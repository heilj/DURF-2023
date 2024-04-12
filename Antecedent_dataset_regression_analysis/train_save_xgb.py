from utils import *
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
def read_token():
    with open("top_viral_embedding_len=500.pkl",'rb') as file:
        token_data = pickle.load(file)
    file.close()
    return token_data
token_data = read_token()
data = read_file('top_dao.csv',token_data,True,7,thres=1000)
data = classfy('viral',data)
independent_variables = np.array([item[1] for item in data])
dependent_values = np.array([item[2][0] for item in data])
scalar = RobustScaler()
independent_variables = logt(independent_variables)
dependent_values = logt(dependent_values)
independent_variables = scalar.fit_transform(independent_variables)
rbf_variables = np.array([item[3] for item in data])
print(rbf_variables.shape)
rbf_variables = np.squeeze(rbf_variables, axis=1)
rbf_variables = rbf_variables.transpose(0,2, 1)
data_tensor = torch.tensor(rbf_variables, dtype=torch.float32)
#manage embedding
#use multiple layers to squeeze the size
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
batch_size = 332  # You can adjust the batch size
shuffle = False  # Shuffle the data each epoch
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=16, stride=16, padding=0)
        self.layer2 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=16, stride=16, padding=0)

    def forward(self, x):

        x = nn.ReLU()(self.layer1(x))
        x = nn.ReLU()(self.layer2(x))
        x = x.transpose(1,2)
        x = torch.squeeze(x)
        return x

# Load the encoder
encoder = CNNEncoder()
encoder_state_dict = torch.load('viral_cnn_encoder.pth')
# Update the model's encoder state dictionary keys
for key in list(encoder_state_dict.keys()):
    new_key = key.replace('0.', 'layer1.').replace('2.', 'layer2.')
    encoder_state_dict[new_key] = encoder_state_dict.pop(key)
encoder.load_state_dict(encoder_state_dict)
encoder.to(device)  # Move to the same device as before
encoder.eval()

# Assuming you have a DataLoader named 'dataloader'
encoded_features = []  # List to store encoded features

with torch.no_grad():
    for data in dataloader:
        # If your data is a tuple of (features, labels), use data[0] to get features
        # If your data is just features, use data directly
        inputs = data.to(device)  # Make sure data is on the same device as the encoder

        # Get encoded features
        encoded = encoder(inputs)
        
        # You might want to move the encoded features to the CPU if they are not needed on the GPU
        encoded = encoded.cpu()

        encoded_features.append(encoded)

# Concatenate all encoded features (optional, depending on your requirement)
encoded_features = torch.cat(encoded_features, 0)
print(encoded_features.shape)
print(type(encoded_features))


independent_variables = np.concatenate((independent_variables,encoded_features),axis=1)
X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=0.3, random_state=42)
X_train_scaled = X_train
X_test_scaled = X_test
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
dependent_values = dependent_values.reshape(-1,1)

viral_xgb_params = {'colsample_bytree': 1.0,
                 'gamma': 1, 
                 'learning_rate': 0.1, 
                 'max_depth': 5, 
                 'min_child_weight': 1, 
                 'n_estimators': 40, 
                 'reg_alpha': 1, 
                 'reg_lambda': 1, 
                 'subsample': 0.4}

quality_xgb_params = {'colsample_bytree': 1.0,
                 'gamma': 0, 
                 'learning_rate': 0.1, 
                 'max_depth': 6, 
                 'min_child_weight': 1, 
                 'n_estimators': 120, 
                 'reg_alpha': 0, 
                 'reg_lambda': 1, 
                 'subsample': 1}

xgb_model = XGBRegressor(**viral_xgb_params)
xgb_model.fit(X_train_scaled, y_train)
y_pred = xgb_model.predict(X_test_scaled) #第一次predict y 使用xtest
y_pred = inverselogt(y_pred)
r2_score = xgb_model.score(X_test_scaled, y_test)
print(f"决定系数（R²）：{r2_score}")
y_test = inverselogt(y_test)
y_train = inverselogt(y_train)
# y_pred = scalar.inverse_transform(y_pred)
y_pred_train = xgb_model.predict(X_train_scaled) #第二次predict y 使用xtrain
y_pred_train = inverselogt(y_pred_train)

def compute_mrse(y_pred,y_true):
    count = 0
    mrse = 0
    for i in range(len(y_pred)):
        mrse += (((y_pred[i])/(y_true[i])) - 1)**2
        count += 1
    mrse = mrse/count
    return mrse

mrse = compute_mrse(y_pred,y_test)
mrse_train = compute_mrse(y_pred_train,y_train)
print(f'test mrse:{mrse}')
print(f'train_mrse:{mrse_train}')


