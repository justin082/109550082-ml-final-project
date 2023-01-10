import numpy as np
import pandas as pd
import gc; gc.enable()
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import KNNImputer
import pickle
import gzip

submission = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test.csv', index_col = 'id')

df_test = test
df_test['area'] = test['attribute_2'] * test['attribute_3']
feature = [f for f in test.columns if f.startswith('measurement') or f=='loading']

for code in test.product_code.unique():
    model1 = KNNImputer(n_neighbors=100)
    df_test.loc[df_test.product_code==code, feature] = model1.fit_transform(df_test.loc[df_test.product_code==code, feature])
test = df_test.iloc[test.shape[0]:,:]

features = ['loading', 'attribute_0', 'measurement_17', 'measurement_0', 'measurement_1', 'measurement_2', 'area']

with gzip.open('model.pgz', 'r') as f:
    model = pickle.load(f)

test_preds = model.predict_proba(df_test[features])[:, 1]

submission['failure'] = test_preds
submission.to_csv('submission.csv', index=False)