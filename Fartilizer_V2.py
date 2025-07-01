import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


target_le = LabelEncoder()
train['Fertilizer Label'] = target_le.fit_transform(train['Fertilizer Name'])


cat_cols = ['Soil Type', 'Crop Type']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le


def engineer_features(df):
    df['NPK_sum'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    df['Moisture_Adjusted_N'] = df['Moisture'] / (df['Nitrogen'] + 1)
    return df

train = engineer_features(train)
test = engineer_features(test)

features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous', 'NPK_sum', 'Moisture_Adjusted_N']
X = train[features]
y = train['Fertilizer Label']
X_test = test[features]


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score, hits = 0.0, 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                hits += 1.0
                score += hits / (i + 1.0)
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train), len(target_le.classes_)))
test_preds = np.zeros((len(test), len(target_le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🔁 Fold {fold+1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(target_le.classes_),
        eval_metric='mlogloss',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        tree_method='gpu_hist',       
        predictor='gpu_predictor'      
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=30,
              verbose=False)

    val_probs = model.predict_proba(X_val)
    oof_preds[val_idx] = val_probs

    fold_map3 = mapk(y_val.values, val_probs.argsort(axis=1)[:, -3:][:, ::-1].tolist(), k=3)
    print(f" Fold {fold+1} MAP@3: {fold_map3:.4f}")

    test_preds += model.predict_proba(X_test) / NUM_FOLDS


oof_top3 = oof_preds.argsort(axis=1)[:, -3:][:, ::-1]
cv_score = mapk(y.values, oof_top3.tolist(), k=3)
print(f"\n Overall CV MAP@3: {cv_score:.4f}")
