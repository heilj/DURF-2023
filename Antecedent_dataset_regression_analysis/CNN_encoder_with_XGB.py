import torch
import torch.nn as nn
from utils import *
from torch.utils.data import TensorDataset, DataLoader, Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print(device)

def read_token(type):
    with open(f"top_{type}_embedding_len=500.pkl",'rb') as file:
        token_data = pickle.load(file)
    file.close()
    data = read_file('top_dao.csv',token_data,True,7,thres=100)
    data = classfy(type,data)

    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])
    input = np.array([item[3] for item in data])
    return input

input1 = read_token('viral')
input2 = read_token('quality')
print(input1.shape)
input = np.concatenate((input1,input2),axis=0)
print(input.shape)
input = np.squeeze(input, axis=1)
input = input.transpose(0,2, 1)
print(input.shape)
data_tensor = torch.tensor(input, dtype=torch.float32)
print(data_tensor.shape)
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# Instantiate the dataset
dataset = CustomDataset(data_tensor)

# Create a DataLoader
batch_size = 128  # You can adjust the batch size
shuffle = True  # Shuffle the data each epoch
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()

        # Encoder with the provided convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7,12), stride=(1,12), padding=0),
            nn.ReLU()
            
        )

        self.xgb = 

        

    def forward(self, x):
        encoded = self.encoder(x)
        # print(f'encoded shape {encoded.shape}')
        predicted = 
        # print(f'decoded shape {decoded.shape}')
        # decoded = decoded.transpose(2, 3)
        # print(f'decoded shape {decoded.shape}')
        return decoded

# Create the model and define the loss function and optimizer
model = CNNAutoencoder()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20

for epoch in range(num_epochs):
    for data in dataloader:  # Assuming dataloader is defined and provides input data
        inputs = data.to(device)  # No labels needed
        inputs = inputs.unsqueeze(1)
        print(inputs.shape)
        # Forward pass
        outputs = model(inputs)
        print(outputs.shape)
        # inputs = inputs.transpose(1,2)
        loss = criterion(outputs, inputs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# encoder_state_dict = model.encoder.state_dict()
# torch.save(encoder_state_dict, 'cnn_encoder.pth')



# def extract_features(input_data):
#     encoder.eval()  # Set the encoder to evaluation mode
#     with torch.no_grad():
#         features = encoder(input_data)
#     return features

# # Example: Extract features from new data
# new_data = torch.tensor([...])  # Some new data tensor
# features = extract_features(new_data)


# torch.save(encoder.state_dict(), 'encoder.pth')
