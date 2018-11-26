import torch
import torch.nn as nn
import torch.nn.functional as F
from Devs_vsSum.datasets import KyLoader
from Devs_vsSum.JM import sum_tools as vsum_tools
import numpy as np
import Network
from PtUtils import cuda_model
from torch.autograd import Variable
import pickle as pkl
def convertsoftmaxTo01(input_np):
    # output = np.zeros(input_np.shape[0])

    # output[input_np[:,1]>=input_np[:,0]]=1
    return input_np[:,1]


eval_dataset = 'SumMe'
if eval_dataset == 'TVSum':
    eval_method = 'avg'
else:
    eval_method = 'max'
dataset = KyLoader.loadKyDataset(eval_dataset)
video_frames = KyLoader.getKyVideoFrames(dataset)
dataset_keys =KyLoader.getKyDatasetKeys(dataset)
np.random.seed(0)
permute_ids = np.random.permutation(len(dataset_keys))
nTrain = 20
nTest = 5
train_keys = [dataset_keys[x] for x in permute_ids[:nTrain]]
test_keys = [dataset_keys[x] for x in permute_ids[nTrain:]]



lr = 0.0001
weight_decay = 1e-5
model = Network.vsLSTM(input_size=1024, hidden_size=256, num_layers=1, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                            weight_decay=weight_decay)

gpu_id = 0
multiGpu= False
useCuda = cuda_model.ifUseCuda(gpu_id, multiGpu)
model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)


# training:
n_epochs = 50
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[20, 40])

epoch_train_losses = []
epoch_test_losses = []
epoch_train_F1scores = []
epoch_test_F1scores = []


for epoch_idx in range(n_epochs):
    lr_scheduler.step(epoch_idx)
    model.train()
    average_score = 0
    train_idx = 0
    average_loss = 0
    for video_idx, s_key in enumerate(train_keys):
        video_features = dataset[s_key]['features'][...]
        full_user_summary = dataset[s_key]['user_summary'][...]
        positions = dataset[s_key]['picks'][...]
        user_summary = full_user_summary[:, positions]
        assert user_summary.shape[1] == video_features.shape[0], "Size of lables ({:d}) is not the same as the size of features ({:d})".format(user_summary.shape[1], video_features.shape[0])
        n_users = user_summary.shape[0]
        video_features = torch.from_numpy(video_features).type(torch.FloatTensor)
        user_summary = torch.from_numpy(user_summary).type(torch.LongTensor)

        pdefined_NFPS = dataset[s_key]['n_frame_per_seg'][...].tolist()
        pdefinedCPS = dataset[s_key]['change_points'][...]
        n_frames = dataset[s_key]['n_frames'][()]

        if useCuda:
            video_features = video_features.cuda()
            user_summary = user_summary.cuda()

        video_features = video_features.unsqueeze(1)

        for user_idx in range(n_users):
            s_user_summary = user_summary[user_idx, :]
            input_var = Variable(video_features)
            target_var = Variable(s_user_summary)
            _, preds, _ = model(input_var, useCuda=useCuda)
            preds = preds.squeeze(1)
            loss = criterion(preds, target_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = F.softmax(preds,dim=1)
            output = convertsoftmaxTo01(probs.data.cpu().numpy())
            machine_summary = vsum_tools.generate_summary(output, pdefinedCPS, n_frames, pdefined_NFPS, positions)
            s_F1_score, _, _ = vsum_tools.evaluate_summary(machine_summary, full_user_summary, eval_method)
            s_loss = loss.data.cpu().numpy()[0]
            train_idx += 1
            average_score += s_F1_score
            average_loss += s_loss
            # print "epoch: {:d}\tvideo:{:d},user:{:d}: Loss: {:.4f}({:.4f})\t Summary Performance: {:.4f}({:.4f})".format(epoch_idx, video_idx, user_idx, s_loss, average_loss/train_idx, s_F1_score, average_score/train_idx)
    print("---- Train Summary: Epoch {:d}, LR: {:.6f}, Epoch Loss: {:.4f}, Epoch Precision: {:.4f}".format(epoch_idx, optimizer.param_groups[0]['lr'], average_loss/(train_idx), average_score/(train_idx)))
    epoch_train_F1scores.append(average_score/train_idx)
    epoch_train_losses.append(average_loss/train_idx)


# test

    average_score = 0
    test_idx = 0
    average_loss = 0
    model.eval()
    for video_idx, s_key in enumerate(test_keys):
        video_features = dataset[s_key]['features'][...]
        full_user_summary = dataset[s_key]['user_summary'][...]
        positions = dataset[s_key]['picks'][...]
        user_summary = full_user_summary[:, positions]
        assert user_summary.shape[1] == video_features.shape[0], "Size of lables ({:d}) is not the same as the size of features ({:d})".format(user_summary.shape[1], video_features.shape[0])
        n_users = user_summary.shape[0]
        video_features = torch.from_numpy(video_features).type(torch.FloatTensor)
        user_summary = torch.from_numpy(user_summary).type(torch.LongTensor)

        pdefined_NFPS = dataset[s_key]['n_frame_per_seg'][...].tolist()
        pdefinedCPS = dataset[s_key]['change_points'][...]
        n_frames = dataset[s_key]['n_frames'][()]

        if useCuda:
            video_features = video_features.cuda()
            user_summary = user_summary.cuda()

        video_features = video_features.unsqueeze(1)
        input_var = Variable(video_features)
        _, preds, _ = model(input_var, useCuda=useCuda)
        preds = preds.squeeze(1)

        probs = F.softmax(preds, dim=1)

        video_loss = 0

        for user_idx in range(n_users):
            s_user_summary = user_summary[user_idx, :]
            target_var = Variable(s_user_summary)
            loss = criterion(preds, target_var)
            s_loss = loss.data.cpu().numpy()[0]
            average_loss += s_loss
            video_loss += s_loss
            test_idx += 1

        output = convertsoftmaxTo01(probs.data.cpu().numpy())
        machine_summary = vsum_tools.generate_summary(output, pdefinedCPS, n_frames, pdefined_NFPS, positions)
        s_F1_score, _, _ = vsum_tools.evaluate_summary(machine_summary, full_user_summary, eval_method)
        average_score += s_F1_score
        # print "TEST: epoch: {:d}\tvideo:{:d}: Loss: {:.4f}({:.4f})\t Summary Performance: {:.4f}({:.4f})".format(epoch_idx, video_idx, video_loss/(n_users), average_loss/test_idx, s_F1_score, average_score/(video_idx+1))
    print("---- Test Summary: Epoch {:d}, Epoch Loss: {:.4f}, Epoch Precision: {:.4f}".format(epoch_idx, average_loss/(test_idx), average_score/(len(test_keys))))
    epoch_test_F1scores.append(average_score/len(test_keys))
    epoch_test_losses.append(average_loss/test_idx)

stats= {}
stats['train_loss']= epoch_train_losses
stats['train_score']=epoch_train_F1scores
stats['test_loss']=epoch_test_losses
stats['test_score']=epoch_test_F1scores

output_file = open('state.pkl', 'wb')
pkl.dump(stats, output_file)


