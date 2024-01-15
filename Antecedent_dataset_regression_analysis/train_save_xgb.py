from utils import *

def read_token():
    with open("top_quality_embedding_len=500.pkl",'rb') as file:
        token_data = pickle.load(file)
    file.close()
    return token_data
token_data = read_token()
data = read_file('top_dao.csv',token_data,True,7,thres=1000)
independent_variables = np.array([item[1] for item in data])
dependent_values = np.array([item[2][0] for item in data])
scalar = RobustScaler()
independent_variables = logt(independent_variables)
dependent_values = logt(dependent_values)
independent_variables = scalar.fit_transform(independent_variables)
rbf_variables = np.array([item[3] for item in data])
print(rbf_variables.shape)
#manage embedding
#use multiple layers to squeeze the size

independent_variables = np.concatenate((independent_variables,rbf_variables),axis=1)
X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=0.3, random_state=42)
X_train_scaled = X_train
X_test_scaled = X_test
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
dependent_values = dependent_values.reshape(-1,1)

xgb_params = {'colsample_bytree': 0.5,
                 'gamma': 0, 
                 'learning_rate': 0.03, 
                 'max_depth': 8, 
                 'min_child_weight': 6, 
                 'n_estimators': 130, 
                 'reg_alpha': 0.1, 
                 'reg_lambda': 0, 
                 'subsample': 0.7}

xgb_model = XGBRegressor(**xgb_params)
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
mrse = 0 
count = 0 
count2 = 0
for i in range(len(y_test)):
    mrse += (((y_pred[i])/(y_test[i])) - 1)**2
    count2 += 1
print('----------------------------------------')

print(count2)
mrse = mrse/count2

count = 0 
count2 = 0
mrse_train = 0
for i in range(len(y_train)):
    mrse_train += (((y_pred_train[i])/(y_train[i])) - 1)**2
    count2 += 1
print('----------------------------------------')

print(count2)
mrse_train = mrse_train/count2