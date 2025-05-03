import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

base_path = os.path.dirname(os.path.abspath(__file__)) 
data_path = os.path.join(base_path, '../Data/diabetes_dataset_processed.csv')
df = pd.read_csv(data_path)

le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])
X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: scaler + XGBClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Có thể bỏ nếu dùng tree-based model, nhưng để đồng bộ với SVC
    ('classifier', XGBClassifier(
        objective='multi:softmax',
        num_class=13,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Grid tham số cho XGBoost
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 6, 10],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
    'classifier__scale_pos_weight': [1, 2, 5]  
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# scoring='precision_macro' để không bỏ sót người bệnh
# scoring='recall_macro' để không muốn dự đoán sai người khỏe mạnh là bị bệnh
# scoring='f1_macro' để cân bằng giữa precision và recall
best_model = GridSearchCV(pipeline, param_grid, cv=cv, scoring='precision_macro', n_jobs=-1)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

model_path = os.path.join(base_path, './Built_model/xgboost_model.pkl')
joblib.dump(best_model, model_path)