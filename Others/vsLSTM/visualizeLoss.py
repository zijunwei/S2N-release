import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

data = pkl.load(open('state.pkl', 'rb'))
ax = plt.subplot(2,1,1)
train_loss = data['train_loss']
test_loss = data['test_loss']
train_score = data['train_score']
test_score = data['test_score']
plt.plot(range(len(train_loss)), train_loss, 'r', label='Train')
plt.plot(range(len(test_loss)), test_loss, 'b', label='Test')
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=2, shadow=True, title="Legend", fancybox=True)
ax.set_title('Loss')

ax = plt.subplot(2,1,2)
ax.plot(range(len(train_score)), train_score, 'r')
ax.plot(range(len(test_score)), test_score, 'b')
ax.set_title('F1 Score')
# plt.show()
plt.savefig('state.PNG')
plt.close()
print "DEBUG"