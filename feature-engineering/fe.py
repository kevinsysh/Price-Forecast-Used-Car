import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

base_dir = os.path.abspath('.')

## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
train_data = pd.read_csv(base_dir+'/data/used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv(base_dir+'/data/used_car_testA_20200313.csv', sep=' ')
train_data[['bodyType', 'fuelType', 'gearbox']] = train_data[[
    'bodyType', 'fuelType', 'gearbox']].fillna(-1)
test_data[['bodyType', 'fuelType', 'gearbox']] = test_data[[
    'bodyType', 'fuelType', 'gearbox']].fillna(-1)

feature_cols = ['notRepairedDamage','brand','regDate','gearbox', 'power', 'offerType', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3',
                'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']

# bodyType
enc_bodyType = preprocessing.OneHotEncoder()
enc_bodyType.fit(train_data['bodyType'].values.reshape(-1, 1))
bodyType = enc_bodyType.transform(
    train_data['bodyType'].values.reshape(-1, 1)).toarray()
bodyType_column = ['bt_-1', 'bt_0', 'bt_1',
                   'bt_2', 'bt_3', 'bt_4', 'bt_5', 'bt_6', 'bt_7']
train_bodyType = pd.DataFrame(data=bodyType, columns=bodyType_column)
train_bodyType = pd.concat([train_data['bodyType'], train_bodyType], axis=1)

# fuelType
enc_fuelType = preprocessing.OneHotEncoder()
enc_fuelType.fit(train_data['fuelType'].values.reshape(-1, 1))
fuelType = enc_fuelType.transform(
    train_data['fuelType'].values.reshape(-1, 1)).toarray()
fuelType_column = ['ft_-1', 'ft_0', 'ft_1',
                   'ft_2', 'ft_3', 'ft_4', 'ft_5', 'ft_6']
train_fuelType= pd.DataFrame(data=fuelType, columns=fuelType_column)
train_fuelType = pd.concat([train_data['fuelType'], train_fuelType], axis=1)

# regDate
train_data.regDate =np.log(train_data.regDate.max() -train_data.regDate+1)

# notRepairedDamage
train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

x_data = train_data[feature_cols]
y_data = train_data['price']

data = pd.concat([y_data, x_data, train_bodyType, train_fuelType], axis=1)
data.to_csv(base_dir+'/data/train.csv', mode='w+', index=False)

# test_data process
saleId = test_data['SaleID']

bodyType = enc_bodyType.transform(test_data['bodyType'].values.reshape(-1, 1)).toarray()
test_bodyType = pd.DataFrame(data=bodyType, columns=bodyType_column)

fuelType = enc_fuelType.transform(test_data['fuelType'].values.reshape(-1, 1)).toarray()
test_fuelType = pd.DataFrame(data=fuelType, columns=fuelType_column)

test_data = test_data[feature_cols]
test_data = pd.concat([saleId, test_data, test_bodyType,test_fuelType], axis=1)
test_data.to_csv(base_dir+'/data/test.csv', mode='w+', index=False)
