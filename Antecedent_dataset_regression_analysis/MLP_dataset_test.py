import datasets
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm

from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import *
from sklearn.linear_model import Ridge
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print(device)

batch_size = 32

# Load dataset
dataset = datasets.load_from_disk("top_quality_dataset_500")
dataset = dataset.map(lambda example: {"x": example["x"][7:], "z": example["x"][:7], "y": example["y"]})
train_x = dataset['train']['z']
train_y = dataset['test']['y']
model = Ridge()
print(train_x[1])