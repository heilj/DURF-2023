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
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print(device)

batch_size = 32

# Load dataset
dataset = datasets.load_from_disk("top_quality_dataset_500")
dataset = dataset.map(lambda example: {"x": example["x"][7:], "z": example["x"][:7], "y": inverselogt(example["y"])})
print(dataset['train']['x'])
print('\n')
print(dataset['train']['y'])
# Load tokenizer and BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
for param in bert_model.parameters():
    param.requires_grad = False
#define loss

def CustomLoss(my_outputs, my_labels):
    my_outputs = torch.expm1(my_outputs)
    my_labels = torch.expm1(my_labels)
    my_outputs = ((my_outputs/my_labels) - 1)**2
    my_outputs = torch.mean(my_outputs)
    return my_outputs


# Tokenize function
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

def rbf_kernel(x1, x2, length_scale=50.0):
    diff = x1 - x2
    return torch.exp(-torch.sum(diff**2) / (2 * length_scale**2))


dataset.set_format("torch")

# Define a custom classifier on top of BERT encoder

def res(input_channel):
    block = nn.Sequential(
          nn.Conv2d(input_channel,input_channel,3, padding = 1),
          nn.BatchNorm2d(input_channel),
          nn.ReLU(),
          nn.Conv2d(input_channel,input_channel,3, padding = 1),
          nn.BatchNorm2d(input_channel),
        )
    
    return nn.Sequential(*block)

def conv_block(input_channel, output_channel, filter_size = 3,padding = 1):
    block = nn.Sequential(
          nn.Conv2d(input_channel,output_channel,filter_size, padding = padding),
          nn.BatchNorm2d(output_channel),
          nn.ReLU(),
        )
    
    return nn.Sequential(*block)


class CustomClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomClassifier, self).__init__()
        self.bert = bert_model

        self.layer1 = nn.Linear(bert_model.config.hidden_size, 500)
        self.layer2 = nn.Linear(500, 100)
        self.layer3 = nn.Linear(100, 1)
        self.layer4 = nn.Linear(507, 1)

    def forward(self, input_ids, seven_views, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        print(outputs.shape)
        outputs = F.relu(self.layer1(outputs))
        outputs = F.relu(self.layer2(outputs))
        outputs = self.layer3(outputs)
        print(outputs.shape)
        result = torch.cat((outputs.view(-1, 500), seven_views), dim=1)
        outputs = self.layer4(result)
        return outputs

# Create an instance of the custom classifier
custom_model = CustomClassifier(bert_model)
# Move the model to the desired device
custom_model.to(device)
# Create optimizer and scheduler
optimizer = AdamW(custom_model.parameters(), lr=5e-5,  weight_decay=1.0)
num_epochs = 20
num_training_steps = num_epochs * len(dataset["train"]) // batch_size  # Adjust batch size
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

# Training loop
progress_bar = tqdm(range(num_training_steps))
# custom_model.train()

# for epoch in range(num_epochs):
#     for batch in DataLoader(dataset["train"], shuffle=True, batch_size=batch_size):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = custom_model(batch["x"].int(), batch["z"]).to(device) 
#         loss = CustomLoss(outputs, batch["y"].float())      
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)
#     #lr_scheduler.step()
#     my_lr = lr_scheduler.get_last_lr()[0]
#     print("loss:", loss, "lr", my_lr)

# PATH = '/Users/gqs/Documents/DURF/DURF-2023/Antecedent_dataset_regression_analysis/model_quality.pth'
# torch.save(custom_model.state_dict(), PATH)
# Evaluation
# metric = evaluate.load("accuracy")

# state_dict = torch.load('model.pth',map_location=torch.device('mps') )
# custom_model.load_state_dict(state_dict)


total_loss = 0.0
total_batches = 0
custom_model.load_state_dict(torch.load('model_quality.pth'))
custom_model.eval()
# Iterate through the test data
with torch.no_grad():
    for batch in DataLoader(dataset["test"], batch_size=batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward pass
        outputs = custom_model(batch["x"].int(), batch["z"]).to(device) 
        # Calculate the loss
        loss = CustomLoss(outputs, batch["y"].float())     
        # Accumulate the loss and update the number of batches
        total_loss += loss.item()
        total_batches += 1

# Calculate the average loss over all batches
average_loss = total_loss / total_batches
# Print or use the average loss as needed
print("Average Test Loss:", average_loss)
