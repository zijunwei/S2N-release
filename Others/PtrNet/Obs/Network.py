import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from PtUtils import cuda_model
from torch.nn import Parameter

class ptrLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ptrLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True)

        self.attWRef = nn.linear(hidden_size*2, hidden_size*2)
        self.attWq = nn.linear(hidden_size*2, hidden_size*2)
        self.attV = Parameter(hidden_size*2)

    def forward(self, x, h0=None, c0=None, useCuda=False):
        # Set initial states
        h0 = h0 or torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = c0 or torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        if useCuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        h0 = Variable(h0)  # 2 for bidirection
        c0 = Variable(c0)

        # Forward propagate RNN
        lstm_out, _ = self.encoder(x, (h0, c0))

        # Decode hidden state of last time step
        video_out = self.fc_videoscore(lstm_out[:, -1, :])
        frame_out = self.fc_framescore(lstm_out)
        # frame_outs = self.fc_framescore(lstm_out[])
        return video_out, frame_out, lstm_out


if __name__ == '__main__':
    # Hyper Parameters
    sequence_length = 28
    input_size = 28
    hidden_size = 128
    num_layers = 1
    num_classes = 10
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.003

    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    rnn = ptrLSTM(input_size, hidden_size, num_layers, num_classes)
    gpu_id = 0
    multiGpu = False
    useCuda = cuda_model.ifUseCuda(gpu_id, multiGpu)
    rnn = cuda_model.convertModel2Cuda(rnn, gpu_id=gpu_id, multiGpu=multiGpu)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, sequence_length, input_size))
            labels = Variable(labels)
            if useCuda:
                images = images.cuda()
                labels = labels.cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs, _ = rnn(images, useCuda=useCuda)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        if useCuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images.view(-1, sequence_length, input_size))
        outputs, _ = rnn(images, useCuda=useCuda)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Model
    # torch.save(rnn.state_dict(), 'rnn.pkl')