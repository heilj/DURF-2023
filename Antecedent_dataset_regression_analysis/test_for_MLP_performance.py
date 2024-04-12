from utils import *
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

data = read_file('top_dao.csv',None,False,7,thres=1000)
data = classfy('viral',data)
independent_variables = np.array([item[1] for item in data])
dependent_values = np.array([item[2][0] for item in data])
scalar = RobustScaler()
independent_variables = logt(independent_variables)
dependent_values = logt(dependent_values)
independent_variables = scalar.fit_transform(independent_variables)
X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=0.3, random_state=42)
X_train_scaled = X_train
X_test_scaled = X_test
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
dependent_values = dependent_values.reshape(-1,1)

mlp = MLPRegressor(random_state=1,
                    max_iter=500,
                    activation='relu',
                    early_stopping=True,
                    solver='sgd',
                    batch_size=300,
                    hidden_layer_sizes=[100,50]
                    )
mlp.fit(X_train_scaled,y_train)

y_pred_test = mlp.predict(X_test)
y_pred_test = inverselogt(y_pred_test)
y_test = inverselogt(y_test)
def compute_mrse(y_pred,y_true):
    count = 0
    mrse = 0
    for i in range(len(y_pred)):
        mrse += (((y_pred[i])/(y_true[i])) - 1)**2
        count += 1
    mrse = mrse/count
    return mrse

mrse = compute_mrse(y_pred_test,y_test)
print(mrse)