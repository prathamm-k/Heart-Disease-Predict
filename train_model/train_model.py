import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import xgboost as xgb
import os

# Ensure output directory exists
os.makedirs('train_model', exist_ok=True)

# Only use these top 5 features
top_features = ['cp', 'ca', 'thal', 'thalach', 'oldpeak']

# Load dataset
train_data_path = os.path.join('train_model', 'heart.csv')
df = pd.read_csv(train_data_path)
X = df[top_features]
y = df['target']

# Preprocessing pipeline
numeric_features = ['thalach', 'oldpeak']
categorical_features = ['cp', 'ca', 'thal']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Models to compare
models = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
}

# Hyperparameter grids (smaller for speed)
param_grids = {
    'RandomForest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 5]
    },
    'XGBoost': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.01, 0.1]
    }
}

best_score = 0
best_name = None
best_pipeline = None

for name, model in models.items():
    print(f'\nTuning {name}...')
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    search = RandomizedSearchCV(pipe, param_grids[name], n_iter=5, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1, verbose=2)
    search.fit(X, y)
    score = search.best_score_
    print(f'{name} ROC-AUC: {score:.4f}')
    if score > best_score:
        best_score = score
        best_name = name
        best_pipeline = search.best_estimator_

print(f'\nBest model: {best_name} with ROC-AUC: {best_score:.4f}')

# Save the best model pipeline
joblib.dump(best_pipeline, os.path.join('train_model', 'heart_model_pipeline.joblib'))

# Save feature names used
joblib.dump(top_features, os.path.join('train_model', 'model_features.joblib'))

# Save feature importances if available
if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
    importances = best_pipeline.named_steps['model'].feature_importances_
    feature_importance_df = pd.DataFrame({'feature': top_features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    feature_importance_df.to_csv(os.path.join('train_model', 'feature_importances.csv'), index=False) 