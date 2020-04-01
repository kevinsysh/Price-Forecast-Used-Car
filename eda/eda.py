import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

baseDir = os.path.abspath('.')

## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
Train_data = pd.read_csv(baseDir+'/data/used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv(baseDir+'/data/used_car_testA_20200313.csv', sep=' ')

# print('train data shape {}'.format(Train_data.shape))
# print('test data shape {}'.format(TestA_data.shape))

# print(Train_data.info())
# print(TestA_data.info())

# print(Train_data.describe())

print(Train_data.regDate.max())
Train_data.regDate =np.log(Train_data.regDate.max() -Train_data.regDate+1)

plt.scatter(Train_data.regDate, Train_data.price)
plt.ylabel("price")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title("regDate")

# Train_data.regDate-Train_data.regDate

# plt.show()
categorical_features = ['name','model','brand','bodyType','fuelType','gearbox',
                         'power','kilometer','notRepairedDamage','regionCode']

# for fea in categorical_features:   
#     print(fea) 
#     print(Train_data[fea].nunique())

Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

print(Train_data['notRepairedDamage'])
print(Train_data['notRepairedDamage'].value_counts())