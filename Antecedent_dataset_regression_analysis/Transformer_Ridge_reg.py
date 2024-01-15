import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

CLASS1 = 'viral'
CLASS2 = 'quality'
CLASS3 = 'memoryless'
CLASS = CLASS1

# Load BERT model
bert_model = AutoModel.from_pretrained("bert-base-cased")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

#RBF
class CustomRBF(nn.Module):
    def __init__(self, kernel, batch_size = batch_size):
        super(CustomRBF, self).__init__()
        self.kernel = kernel
        self.linear = nn.Linear(1,1)

    def forward(self, X, centroid):
        n_samples = X.size(0)
        #print(X[0].size(), centroid.size())
        K = torch.zeros((n_samples, 1)).to(device)
        for i in range(n_samples):
            K[i] = self.kernel(X[i], centroid)
        #print(K.size())
        output = self.linear(K)

        return output.squeeze()

class TransformerRidgeRegressor(nn.Module):
    def __init__(self, model):
        super(TransformerRidgeRegressor, self).__init__()
        self.transformer = model


    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return transformer_output.pooler_output
    


def train_transformer_ridge(transformer_ridge, data_loader, ridge_regressor):
    transformer_ridge.eval()
    embeddings = []
    targets = []
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        with torch.no_grad():
            output = transformer_ridge(input_ids=input_ids, attention_mask=attention_mask)
        embeddings.append(output.numpy())
        targets.append(labels.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    ridge_regressor.fit(embeddings, targets)

def evaluate_model(transformer_ridge, data_loader, ridge_regressor):
    transformer_ridge.eval()
    predictions = []
    true_labels = []
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        with torch.no_grad():
            output = transformer_ridge(input_ids=input_ids, attention_mask=attention_mask)
        pred = ridge_regressor.predict(output.numpy())
        predictions.extend(pred)
        true_labels.extend(labels.numpy())
    
    mse = mean_squared_error(true_labels, predictions)
    return mse

# Example usage
# tokenizer = AutoTokenizer.from_pretrained('model_name')
# dataset_encoded = ...
# data_loader = DataLoader(dataset_encoded, batch_size=32, shuffle=True)
# model = TransformerRidgeRegressor('model_name', freeze_transformer=True)
# ridge = Ridge()
# train_transformer_ridge(model, data_loader, ridge)
# mse = evaluate_model(model, data_loader, ridge)
# print(f"Mean Squared Error: {mse}")
