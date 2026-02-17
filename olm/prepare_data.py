import torch
from torchvision import datasets

SEQ_LEN = 784
TRAIN_SIZE = 5000
TEST_SIZE = 200
SEED = 42

# load MNIST data
train_dataset = datasets.MNIST(root='./data', train=True, transform=None, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=None, download=True)

torch.manual_seed(SEED)

# Select the train/test splits
train_data, train_labels = train_dataset.data, train_dataset.targets
test_data, test_labels = test_dataset.data, test_dataset.targets

perm = torch.randperm(train_data.shape[0])
train_data, train_labels = train_data[perm][:TRAIN_SIZE], train_labels[perm][:TRAIN_SIZE]

perm = torch.randperm(test_data.shape[0])
test_data, test_labels = test_data[perm][:TEST_SIZE], test_labels[perm][:TEST_SIZE]

# Deterministic binarization (pixel = 1 if intensity > 0.5 else 0)
train_data_det = (train_data > 0.5).float().view(-1, SEQ_LEN)
test_data_det = (test_data > 0.5).float().view(-1, SEQ_LEN)
torch.save({
    "train_data": train_data_det, 
    "train_labels": train_labels,
    "test_data": test_data_det, 
    "test_labels": test_labels
}, "det_dataset.pt")

# Stochastic binarization (pixel ~ Bernoulli(intensity))
train_data_stoch = torch.bernoulli(train_data.float() / 255.0).float().view(-1, SEQ_LEN)
test_data_stoch = torch.bernoulli(test_data.float() / 255.0).float().view(-1, SEQ_LEN)
torch.save({
    "train_data": train_data_stoch, 
    "train_labels": train_labels,
    "test_data": test_data_stoch, 
    "test_labels": test_labels
}, "stoch_dataset.pt")