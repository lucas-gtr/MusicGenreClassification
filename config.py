import torch

batch_size = 8
use_augmentation = False
epoch_number = 30
num_chunks = 1
sample_rate = 22050
sample_length = 29
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

