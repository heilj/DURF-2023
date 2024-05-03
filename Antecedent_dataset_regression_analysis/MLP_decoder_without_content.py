import datasets
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import *
from sklearn.decomposition import PCA

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

batch_size = 32

pca = PCA(n_components=1)

# Load dataset
dataset = datasets.load_from_disk("top_viral_onehot")
X_train = np.array(dataset['train']['x'])[:,:7]
print(X_train.shape)
content_train = np.array(dataset['train']['x'])[:,7:]
print(content_train.shape)
y_train = np.array(dataset['train']['y']).reshape(-1,1)
X_test = np.array(dataset['test']['x'])[:,:7]
content_test = np.array(dataset['test']['x'])[:,7:]
y_test = np.array(dataset['test']['y']).reshape(-1,1)
x_scalar = StandardScaler()
y_scalar = StandardScaler()

pca_train = pca.fit_transform(content_train)
pca_test = pca.transform(content_test)
print(X_test.shape)
print(pca_test.shape)
X_train = np.concatenate((X_train,pca_train),axis=1)
X_test  = np.concatenate((X_test,pca_test),axis=1)

X_train_standardized = x_scalar.fit_transform(X_train)
X_mean = x_scalar.mean_
X_std = x_scalar.scale_
y_train_standardized = y_scalar.fit_transform(y_train)
y_mean = torch.tensor(y_scalar.mean_)
y_std = torch.tensor(y_scalar.scale_)
# Normalize the data:

X_test_standardized = x_scalar.transform(X_test)

y_test_standardized = y_scalar.transform(y_test)

train_dict = {'x': X_train_standardized,'y': y_train_standardized}
test_dict = {'x': X_test_standardized, 'y': y_test_standardized}
# dataset['train']['x'] = X_train_standardized
# dataset['train']['y'] = y_train_standardized
# dataset['test']['x']  = X_test_standardized
# dataset['test']['y']  = y_test_standardized
dataset['train'] = datasets.Dataset.from_dict(train_dict)
dataset['test'] = datasets.Dataset.from_dict(test_dict)
# Load tokenizer and BERT model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# bert_model = AutoModel.from_pretrained("bert-base-uncased")
# for param in bert_model.parameters():
#     param.requires_grad = False
#define loss

def save_loss_plot(train_loss, valid_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss[10:], color='orange', label='train loss')
    plt.plot(valid_loss[10:], color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.yscale('log')
    # plt.title(f'{summary}')
    plt.legend()
    plt.savefig(f'model_plt/loss_nn_regression.jpg')
    # plt.savefig(f'{path}{summary}.jpg')
    # plt.show()

def CustomLoss(my_outputs, my_labels):
    my_outputs *= y_std
    my_outputs += y_mean
    my_labels *= y_std
    my_labels += y_mean
    my_outputs = torch.expm1(my_outputs)
    my_labels = torch.expm1(my_labels)
    my_outputs = ((my_outputs/my_labels) - 1)**2
    my_outputs = torch.mean(my_outputs)
    return my_outputs


dataset.set_format("torch")

# Define a custom classifier on top of BERT encoder

class CustomClassifier(nn.Module):
    def __init__(self):
        super(CustomClassifier, self).__init__()
        

        self.layer1 = nn.Linear(8, 50)
        # self.layer2 = nn.Linear(200, 100)
        # self.layer3 = nn.Linear(100, 200)
        # self.layer4 = nn.Linear(200, 100)
        # self.layer2 = nn.Linear(100, 50)
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25, 16)
        self.layer4 = nn.Linear(16, 8)
        # self.layer9 = nn.Linear(8,8)
        self.layer5 = nn.Linear(8, 1)
        self.drop = nn.Dropout(p=0.5)


    def forward(self, inputs):
        # inputs = inputs[:,:7]
        outputs = F.sigmoid(self.layer1(inputs))
        # outputs = self.drop(outputs)
        outputs = F.sigmoid(self.layer2(outputs))
        # outputs = self.drop(outputs)
        outputs = F.sigmoid(self.layer3(outputs))
        # outputs = self.drop(outputs)
        outputs = F.sigmoid(self.layer4(outputs))
        # outputs = F.sigmoid(self.layer5(outputs))
        # outputs = F.sigmoid(self.layer6(outputs))
        # outputs = F.sigmoid(self.layer7(outputs))
        # outputs = F.sigmoid(self.layer8(outputs))
        # outputs = F.sigmoid(self.layer9(outputs))
        outputs = (self.layer5(outputs))
        return outputs

mse = nn.MSELoss()

# Create an instance of the custom classifier
custom_model = CustomClassifier()
# Move the model to the desired device
custom_model.to(device)
# Create optimizer and scheduler

optimizer = AdamW(custom_model.parameters(), lr=5e-3,  weight_decay=0.1)
num_epochs = 3000
num_training_steps = num_epochs * len(dataset["train"]) // batch_size  # Adjust batch size
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

# Training loop
progress_bar = tqdm(range(num_training_steps))
# custom_model.load_state_dict(torch.load("nn_quality_lr_1e-3_pac=1"))
custom_model.train()
train_dataload = DataLoader(dataset["train"], shuffle=True, batch_size=batch_size)
test_dataload = DataLoader(dataset["test"], shuffle=True, batch_size=batch_size)
train_loss = []
valid_loss = []
for epoch in range(num_epochs):
    t_batch = 0
    v_batch = 0
    t_epoch_loss = 0
    v_epoch_loss = 0
    for batch in train_dataload:
        t_batch += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = custom_model(batch["x"]).to(device) 
        loss = CustomLoss(outputs, batch["y"].float()) 
        # print("loss in batch training:", loss)
        t_epoch_loss +=  loss.item()   
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    for batch in test_dataload:
        v_batch += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward pass
        outputs = custom_model(batch["x"]).to(device) 
        # Calculate the loss
        loss = CustomLoss(outputs, batch["y"].float())     
        # Accumulate the loss and update the number of batches
        v_epoch_loss += loss.item()
    v_epoch_loss = v_epoch_loss/ v_batch
    t_epoch_loss = t_epoch_loss / t_batch

 
    valid_loss.append(v_epoch_loss) 
    train_loss.append(t_epoch_loss)
    #lr_scheduler.step()
    my_lr = lr_scheduler.get_last_lr()[0]
    print("loss:", t_epoch_loss, "lr", my_lr)
    print("valid loss:", v_epoch_loss, "lr", my_lr)
torch.save(custom_model.state_dict(), "nn_model_lr_1e-3_temp")
save_loss_plot(train_loss, valid_loss)

# PATH = '/Users/gqs/Documents/DURF/DURF-2023/Antecedent_dataset_regression_analysis/model_quality2.pth'
# torch.save(custom_model.state_dict(), PATH)
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
        outputs = custom_model(batch["x"]).to(device) 
        # print(outputs)
        # Calculate the loss
        loss = CustomLoss(outputs, batch["y"].float())     
        # Accumulate the loss and update the number of batches
        total_loss += loss.item()
        total_batches += 1

# Calculate the average loss over all batches
average_loss = total_loss / total_batches
# Print or use the average loss as needed
print("Average Test Loss:", average_loss)
