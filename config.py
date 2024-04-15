import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_percentage = 0.7
val_percentage = 0.2
test_percentage = 0.1

use_augmentation = False

batch_size = 8
epoch_number = 30

num_chunks = 5

sample_rate = 22050
sample_length = 29

