import os
import sys
baseDir = os.path.abspath('.')
sys.path.append(baseDir)
from utils.stat import stat

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from sklearn.svm import SVR 
from sklearn.preprocessing import MinMaxScaler

Train_data = pd.read_csv(baseDir+'/data/train.csv')
X_data = Train_data.iloc[:, 1:]
Y_data = Train_data.iloc[:, 0]
x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.3)

def build_model_svm(x_train, y_train):
    model = SVR(kernel='linear',cache_size=2000)
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    model.fit(x_train,y_train)
    return model

def build_model_xgb(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,
                             colsample_bytree=0.9, max_depth=7)  # , objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127, n_estimators=150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm

feature_cols = ['notRepairedDamage','brand','regDate','bodyType','fuelType','gearbox', 'power', 'offerType', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3',
                'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
x_train1=x_train[feature_cols]
x_val1=x_val[feature_cols]

print('Train lgb...')
model_lgb = build_model_lgb(x_train1, np.log(y_train))
val_lgb = np.exp(model_lgb.predict(x_val1))
MAE_lgb = mean_absolute_error(y_val, val_lgb)
print('MAE of val with lgb:', MAE_lgb)
pickle.dump(model_lgb, open('./models/lgb.pickle', 'wb'))

print('Train xgb...')
model_xgb = build_model_xgb(x_train1, np.log(y_train))
val_xgb =np.exp(model_xgb.predict(x_val1))
MAE_xgb = mean_absolute_error(y_val, val_xgb)
print('MAE of val with xgb:', MAE_xgb)
pickle.dump(model_xgb, open('./models/xgb.pickle', 'wb'))


feature_cols = ['brand','regDate','gearbox', 'power', 'offerType', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3',
                'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
bodyType_column = ['bt_-1', 'bt_0', 'bt_1',
                   'bt_2', 'bt_3', 'bt_4', 'bt_5', 'bt_6', 'bt_7']
fuelType_column = ['ft_-1', 'ft_0', 'ft_1',
                   'ft_2', 'ft_3', 'ft_4', 'ft_5', 'ft_6']
feature_cols = feature_cols+bodyType_column+fuelType_column
x_train2 = x_train[feature_cols]
x_val2=x_val[feature_cols]

print('Train lr')
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(x_train2, np.log(y_train))
val_lr= np.exp(model.predict(x_val2))
MAE_lr = mean_absolute_error(y_val, val_lr)
print('MAE of val with val_lr:',MAE_lr)
print(model.coef_)

# print('Train svm...')
# model_svm = build_model_svm(x_train, y_train)
# val_svm = model_svm.predict(x_val)
# MAE_svm = mean_absolute_error(y_val, val_svm)
# print('MAE of val with svm:',MAE_svm)
# pickle.dump(model_svm, open('./models/svm.pickle', 'wb'))


# 这里我们采取了简单的加权融合的方式
weight = [1/3,1/3,1/3]
val_Weighted=np.sum(np.array(weight).reshape(len(weight),1)* np.array([val_lgb,val_xgb,val_lr]),axis=0)
# 由于我们发现预测的最小值有负数，而真实情况下，price为负是不存在的，由此我们进行对应的后修正
val_Weighted[val_Weighted < 0] = 10
print('MAE of val with Weighted ensemble:',
      mean_absolute_error(y_val, val_Weighted))
pickle.dump([(1-MAE_lgb/(MAE_xgb+MAE_lgb)), (1-MAE_xgb /
                                             (MAE_xgb+MAE_lgb))], open('./models/weight.pickle', 'wb'))
