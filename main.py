import os
import sys
import pandas as pd
import numpy as np
import pickle

baseDir = os.path.abspath('.')

sys.path.append(baseDir)
from utils.stat import stat

Test_data = pd.read_csv(baseDir+'/data/test.csv')
X_test = Test_data.iloc[:, 1:]

with open('./models/lgb.pickle','rb') as f: 
    model_lgb_pre = pickle.load(f)

with open('./models/xgb.pickle','rb') as f: 
    model_xgb_pre = pickle.load(f)

with open('./models/weight.pickle','rb') as f: 
    weight = pickle.load(f)

print('Predict lgb...')
subA_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict lgb:')
stat(subA_lgb)

print('Predict xgb...')
subA_xgb = model_xgb_pre.predict(X_test)
print('Sta of Predict xgb:')
stat(subA_xgb)

sub_Weighted = np.sum(np.array(weight).reshape(2,1)* np.array([subA_lgb,subA_xgb]),axis=0)

sub = pd.DataFrame()
sub['SaleID'] = Test_data.SaleID
sub['price'] = sub_Weighted
sub.to_csv(baseDir+'/data/result.csv', mode='w+', index=False)
