import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

base_path = os.path.dirname(os.path.abspath(__file__))

knn_path = os.path.join(base_path, './Built_model/knn_model.pkl')
logisticregression_path = os.path.join(base_path, './Built_model/logisticregression_model.pkl')
randomforest_path = os.path.join(base_path, './Built_model/randomforest_model.pkl')
svm_path = os.path.join(base_path, './Built_model/svm_model.pkl')
xgboost_path = os.path.join(base_path, './Built_model/xgboost_model.pkl')

knn_model = joblib.load(knn_path)
logisticregression_model = joblib.load(logisticregression_path)
randomforest_model = joblib.load(randomforest_path)
svm_model = joblib.load(svm_path)
xgboost_model = joblib.load(xgboost_path)

data_path = os.path.join(base_path, '../Data/diabetes_dataset.csv')

def prediction(user_data):
    df = pd.read_csv(data_path)
    df = pd.concat([df, user_data], ignore_index=True)
    find_result = {str(Target) : 0 for Target in df['Target'].unique()}
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target'])
    list_columns = df.columns.tolist()

    numerical_columns = []

    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        valid_ratio = coerced.notna().mean()
        if valid_ratio >= 1.0:
            numerical_columns.append(col)

    category_columns = list(set(list_columns) - set(numerical_columns) - set(['Target']))
    encoder = OneHotEncoder(sparse_output=False)
    encoded_df = encoder.fit_transform(df[category_columns])
    encoded_df = pd.DataFrame(encoded_df, columns=encoder.get_feature_names_out(category_columns))

    df_processed = pd.concat([df[numerical_columns], encoded_df], axis=1)
    df_processed.astype(float)

    df_cur = df_processed.iloc[[-1]]
    model_columns = xgboost_model.best_estimator_.feature_names_in_
    df_cur = df_cur[model_columns]

    knn_pre = str(le.inverse_transform(knn_model.predict(df_cur))[0])
    logisticregression_pre = str(le.inverse_transform(logisticregression_model.predict(df_cur))[0])
    randomforest_pre = str(le.inverse_transform(randomforest_model.predict(df_cur))[0])
    svm_pre = str(le.inverse_transform(svm_model.predict(df_cur))[0])
    xgboost_pre = str(le.inverse_transform(xgboost_model.predict(df_cur))[0])

    # Kết luận bằng cách dựa vào các kết luận của 5 mô hình và trọng số của chúng
    find_result[knn_pre] += 50
    find_result[logisticregression_pre] += 75
    find_result[randomforest_pre] += 90
    find_result[svm_pre] += 75
    find_result[xgboost_pre] += 90

    result = max(find_result, key=find_result.get)
    return result
