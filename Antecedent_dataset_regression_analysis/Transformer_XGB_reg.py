import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
# import evaluate
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from utils import *
from xgboost import XGBRegressor

CLASS1 = 'viral'
CLASS2 = 'quality'
CLASS3 = 'memoryless'
CLASS = CLASS1

# Load BERT model
bert_model = AutoModel.from_pretrained("bert-base-cased")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print(device)

# Load dataset
dataset = load_from_disk("top_quality_dataset_500")
dataset.set_format("torch")
train_dataset = dataset['train']
test_dataset = dataset['test']

# no need to redo tokenize

#train the xgb on existing data since it can't be trained with bert
# def trained_xgb(params,train_data,y):        
#     xgb_model = XGBRegressor(**params)
#     xgb_model.fit(train_data,y)
#     return xgb_model

#load the pretrained model
xgb_regressor = XGBRegressor()
xgb_regressor.load_model('.json')

attention_mask = torch.ones(1, 508)
# Define a custom regressor on top of BERT encoder
class CustomRegressor(nn.Module):
    def __init__(self,bert_model,trained_xgb):
        super(CustomRegressor,self).__init__()
        self.bert = bert_model
        self.xgb = trained_xgb
        # CNN layers with stride
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, stride=5, padding=0)
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, stride=4, padding=0)

    def forward(self, x, attention_mask=None):
        view_data = x[:7]
        raw_content_data = x[7:508]
        content_data = self.bert(raw_content_data, attention_mask=attention_mask).last_hidden_state 
        # Transpose content_data to match input shape for Conv1d (batch, channels, length)
        content_data = content_data.transpose(1, 2)  # Shape: (batch_size, 768, 500)
        # Apply CNN layers
        content_data = self.conv1(content_data)  # First convolution layer
        content_data = self.conv2(content_data)  # Second
        # Flatten the last two dimensions
        content_data = content_data.view(content_data.size(0), -1)
        content_data = content_data.cpu().detach().numpy()
        independent_variables = np.concatenate((view_data,content_data),axis=1)
        
        regressor = self.xgb
        prediction = regressor.predict(independent_variables)
        return torch.from_numpy(prediction).float().to(device)    


# Create an instance of the custom classifier
custom_model = CustomRegressor(bert_model,)
# Move the model to the desired device
custom_model.to(device)
# Create optimizer and scheduler
optimizer = AdamW(custom_model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataset) // 64  # Adjust batch size
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

# Training loop
progress_bar = tqdm(range(num_training_steps))
custom_model.train()

for epoch in range(num_epochs):
    for batch in DataLoader(train_dataset, shuffle=True, batch_size=64):
        batch = {k: v.to(device) for k, v in batch.items()}
      
        outputs = custom_model(batch["x"])
        outputs.requires_grad = True
        loss = nn.MSELoss()(outputs, batch["y"].float())      
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    #lr_scheduler.step()
    my_lr = lr_scheduler.get_last_lr()[0]
    print("loss:", loss, "lr", my_lr)

# Evaluation
metric = evaluate.load("accuracy")
custom_model.eval()

total_loss = 0.0
total_batches = 0

# Iterate through the test data
with torch.no_grad():
    for batch in DataLoader(test_dataset, batch_size=64):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward pass
        outputs = custom_model(batch["x"])
        # Calculate the loss
        loss = nn.MSELoss()(outputs, batch["y"].float())
        # Accumulate the loss and update the number of batches
        total_loss += loss.item()
        total_batches += 1

# Calculate the average loss over all batches
average_loss = total_loss / total_batches
# Print or use the average loss as needed
print("Average Test Loss:", average_loss)



