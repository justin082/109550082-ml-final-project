import numpy as np
import pandas as pd
import gc; gc.enable()
import warnings
warnings.filterwarnings('ignore')
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from category_encoders import WOEEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
import optuna
from optuna.pruners import PercentilePruner
import pickle
import gzip

RANDOM_SEED = 153
NUM_TRIALS = 100

submission = pd.read_csv('sample_submission.csv')
train = pd.read_csv('train.csv', index_col = 'id') 
test = pd.read_csv('test.csv', index_col = 'id')

target = train['failure'].copy()

model_list = []
model_list.append(0)

def preprocessing(df_train, df_test):
    data = pd.concat([df_train, df_test])
    data['area'] = data['attribute_2'] * data['attribute_3']

    feature = [f for f in data.columns if f.startswith('measurement') or f=='loading']

    for code in data.product_code.unique():
        model1 = KNNImputer(n_neighbors=100)
        data.loc[data.product_code==code, feature] = model1.fit_transform(data.loc[data.product_code==code, feature])

    df_train = data.iloc[:df_train.shape[0],:]
    features = ['loading', 'attribute_0', 'measurement_17', 'measurement_0', 'measurement_1', 'measurement_2', 'area']
    
    return df_train, features

train, features = preprocessing(train, test)

fold_num = 7
length = len(train)//fold_num
train_size = (length//10) * 7
SPLITS = []
last = 0
for fold in range(fold_num):
    a = np.arange(last, last+length)
    b = a[0:train_size]
    c = a[length-train_size:length]
    last = last+length
    fold = []
    fold.append(b)
    fold.append(c)
    SPLITS.append(fold)

preprocessing = make_pipeline(
    make_column_transformer(
        (WOEEncoder(), ['attribute_0']),
        (FunctionTransformer(np.log1p), ['loading']),
        remainder = 'passthrough'
    ),
    RobustScaler()
)

pruner = PercentilePruner(
    percentile = 75,
    n_startup_trials = 15,
    n_warmup_steps = 5,
    interval_steps = 1,
    n_min_trials = 15,
)

def training(sklearn_model):
    scores = np.zeros(len(SPLITS))
    model = make_pipeline(
            clone(preprocessing),
            clone(sklearn_model)
        )
    model_list.append(model)
    best_model = model
    best_score = 0
    for fold, (train_idx, valid_idx) in enumerate(SPLITS):
        X_train = train[features].iloc[train_idx].copy()
        X_valid = train[features].iloc[valid_idx].copy()
        y_train = target.iloc[train_idx].copy()
        y_valid = target.iloc[valid_idx].copy()
        
        model.fit(X_train, y_train)

        valid_preds = model.predict_proba(X_valid)[:,1]
        scores[fold] = roc_auc_score(y_valid, valid_preds)
        if scores[fold] > best_score: 
            best_score = scores[fold]
            best_model = model

    if best_score > model_list[0]:
        model_list[0] = best_score
        model_list[1] = best_model 
    print(f'Worst: {round(np.min(scores), 6)}')
    print(f'Average:{round(np.mean(scores), 6)}')

    return np.mean(scores)

default_params = dict(            
    solver = 'liblinear',
    penalty = 'l1',
    max_iter = 500,
    random_state = RANDOM_SEED,
)

def parameter_search(trials):
    def objective(trial):
        model_params = dict( 
            C = trial.suggest_loguniform(
                "C", 1e-10, 100
            ),
        )
        model = LogisticRegression(
            **default_params,
            **model_params
        )
        return training(model)
    
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(pruner = pruner,direction = "maximize")
    
    study.enqueue_trial({'C': 1.0})
    study.optimize(objective, n_trials=trials)
    return study
study = parameter_search(NUM_TRIALS)

with gzip.GzipFile('model.pgz', 'w') as f:
    pickle.dump(model_list[1], f)