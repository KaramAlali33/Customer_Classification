import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.drop(columns=['id', 'Response'])
y = train['Response']
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1599, random_state=37)
model = LGBMClassifier(
    num_leaves=50,
    max_depth=10,
    learning_rate=0.01,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)
model.fit(X_train, y_train)
y_val_pred = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation ROC-AUC: {roc_auc}')
X_test = test.drop(columns=['id'])
test['Response'] = model.predict_proba(X_test)[:, 1]
submission = test[['id', 'Response']]
submission.to_csv('submission.csv', index=False)
