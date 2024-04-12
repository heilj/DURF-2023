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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print(device)

batch_size = 32

# Load dataset
dataset = datasets.load_from_disk("top_quality_dataset_500")
dataset = dataset.map(lambda example: {"x": example["x"][7:], "z": example["x"][:7], "y": example["y"]})

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


dataset.set_format("torch")

# Define a custom classifier on top of BERT encoder

class CustomClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomClassifier, self).__init__()
        

        self.layer1 = nn.Linear(7, 50)
        self.layer2 = nn.Linear(50, 150)
        self.layer3 = nn.Linear(150, 100)
        self.layer4 = nn.Linear(100, 25)
        self.layer5 = nn.Linear(25, 1)


    def forward(self, seven_views, attention_mask=None):
        
        outputs = F.relu(self.layer1(seven_views))
        outputs = F.relu(self.layer2(outputs))
        outputs = F.relu(self.layer3(outputs))
        outputs = F.relu(self.layer4(outputs))
        outputs = F.relu(self.layer5(outputs))
        return outputs

# Create an instance of the custom classifier
custom_model = CustomClassifier(bert_model)
# Move the model to the desired device
custom_model.to(device)
# Create optimizer and scheduler
optimizer = AdamW(custom_model.parameters(), lr=5e-5,  weight_decay=1.0)
num_epochs = 50
num_training_steps = num_epochs * len(dataset["train"]) // batch_size  # Adjust batch size
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

# Training loop
progress_bar = tqdm(range(num_training_steps))
custom_model.train()

for epoch in range(num_epochs):
    for batch in DataLoader(dataset["train"], shuffle=True, batch_size=batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = custom_model(batch["z"]).to(device) 
        loss = CustomLoss(outputs, batch["y"].float())      
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    #lr_scheduler.step()
    my_lr = lr_scheduler.get_last_lr()[0]
    print("loss:", loss, "lr", my_lr)

PATH = '/Users/gqs/Documents/DURF/DURF-2023/Antecedent_dataset_regression_analysis/model_quality2.pth'
torch.save(custom_model.state_dict(), PATH)
# Evaluation
# metric = evaluate.load("accuracy")

# state_dict = torch.load('model.pth',map_location=torch.device('mps') )
# custom_model.load_state_dict(state_dict)

custom_model.eval()
total_loss = 0.0
total_batches = 0

# Iterate through the test data
with torch.no_grad():
    for batch in DataLoader(dataset["test"], batch_size=batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward pass
        outputs = custom_model(batch["z"]).to(device) 
        print(outputs)
        # Calculate the loss
        loss = CustomLoss(outputs, batch["y"].float())     
        # Accumulate the loss and update the number of batches
        total_loss += loss.item()
        total_batches += 1

# Calculate the average loss over all batches
average_loss = total_loss / total_batches
# Print or use the average loss as needed
print("Average Test Loss:", average_loss)
