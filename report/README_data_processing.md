# Xử Lý Dữ Liệu và Áp Dụng Cho Các Mô Hình

## 📊 Tổng Quan Xử Lý Dữ Liệu

Dự án sử dụng một pipeline xử lý dữ liệu thống nhất để chuẩn bị dữ liệu cho tất cả các mô hình machine learning. Pipeline này đảm bảo tính nhất quán và tái sử dụng dữ liệu đã xử lý.

## 🔄 Pipeline Xử Lý Dữ Liệu

### 1. **Loading Raw Data**

**File**: `src/utils.py` - Function `load_data()`

```python
# Load từ processed data file
df = load_data(file_path=PROCESSED_DATA_FILE, use_cache=True)
```

**Tính năng**:
- **Caching**: Dữ liệu được cache trong memory để tránh reload
- **Type Optimization**: Tự động tối ưu dtypes để giảm memory
- **Validation**: Kiểm tra file tồn tại và không rỗng

**Output**: DataFrame với 70,000 rows, 59 columns

### 2. **Data Preprocessing**

**File**: `src/utils.py` - Function `preprocess_data()`

#### 2.1. **Categorical Encoding**

```python
# OneHotEncoder cho categorical features
encoders = {}
for col in categorical_columns:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[[col]])
    # Tạo feature names: col_name_category
    # Lưu encoder để sử dụng sau
    encoders[col] = encoder
```

**Xử lý**:
- **OneHotEncoding**: Chuyển categorical thành binary features
- **Handle Unknown**: Ignore categories không thấy trong training
- **Sparse Output**: False (dense matrix) để tương thích với sklearn

**Ví dụ**:
```
"Physical Activity" → ["Physical Activity_High", "Physical Activity_Low", "Physical Activity_Medium"]
```

#### 2.2. **Target Encoding**

```python
# LabelEncoder cho target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Target'])
```

**13 Target Classes**:
1. Cystic Fibrosis-Related Diabetes (CFRD)
2. Gestational Diabetes
3. LADA
4. MODY
5. Neonatal Diabetes Mellitus (NDM)
6. Prediabetic
7. Secondary Diabetes
8. Steroid-Induced Diabetes
9. Type 1 Diabetes
10. Type 2 Diabetes
11. Type 3c Diabetes (Pancreatogenic Diabetes)
12. Wolcott-Rallison Syndrome
13. Wolfram Syndrome

#### 2.3. **Feature Engineering**

**Numerical Features**: Giữ nguyên (58 features sau encoding)

**Polynomial Features** (cho Logistic Regression):
- Degree 2: ~1,770 features từ 58 base features
- Feature Selection: SelectKBest (top 250 features)
- Mục đích: Capture interactions giữa features

### 3. **Data Splitting**

**File**: `src/utils.py` - Function `get_prepared_data()`

```python
# Stratified split để giữ class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Đảm bảo class distribution giống nhau
)
```

**Kết quả**:
- **Train Set**: 56,000 samples (80%)
- **Test Set**: 14,000 samples (20%)
- **Stratified**: Tỷ lệ classes giữ nguyên trong train/test

### 4. **Data Caching**

**File**: `src/utils.py` - Global cache variables

```python
# Cache để chia sẻ dữ liệu giữa các models
_processed_data_cache = None
_split_data_cache = None
```

**Lợi ích**:
- **Memory Efficiency**: Chỉ load và preprocess 1 lần
- **Speed**: Các model sau không cần reload data
- **Consistency**: Tất cả models dùng cùng train/test split

## 🎯 Áp Dụng Cho Từng Mô Hình

### 1. **KNN (K-Nearest Neighbors)**

**Preprocessing**:
- ✅ StandardScaler (required - distance-based)
- ✅ Feature Selection: SelectKBest hoặc PCA
- ✅ n_features: 15-40 (tối ưu bởi Optuna)

**Pipeline**:
```
Input → StandardScaler → Feature Selection → KNN Classifier
```

### 2. **Naive Bayes**

**Preprocessing**:
- ✅ StandardScaler (required for GaussianNB)
- ✅ Support multiple variants (Gaussian, Multinomial, etc.)

**Pipeline**:
```
Input → StandardScaler → Naive Bayes Classifier
```

### 6. **LightGBM**

**Preprocessing**:
- ❌ Không cần StandardScaler (tree-based)
- ✅ Direct feature usage
- ✅ Native categorical support

**Pipeline**:
```
Input → LightGBM Classifier
```

### 3. **Random Forest**

**Preprocessing**:
- ❌ Không cần StandardScaler (tree-based)
- ✅ Direct feature usage
- ✅ Bootstrap sampling (0.6-1.0)

**Pipeline**:
```
Input → Random Forest Classifier
```

### 4. **Logistic Regression**

**Preprocessing**:
- ✅ StandardScaler (required - gradient-based)
- ✅ Polynomial Features (optional, degree 2)
- ✅ Feature Selection sau polynomial (top 250)

**Pipeline**:
```
Input → StandardScaler → [Polynomial Features] → [SelectKBest] → Logistic Regression
```

### 5. **XGBoost**

**Preprocessing**:
- ❌ Không cần StandardScaler (tree-based)
- ✅ Direct feature usage
- ✅ Dynamic num_class detection (13 classes)

**Pipeline**:
```
Input → XGBoost Classifier
```

## 🔧 Data Preparation Function

**File**: `src/utils.py` - Function `get_prepared_data()`

```python
def get_prepared_data(use_cache=True):
    """
    Get prepared training data with caching
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, encoders
    """
    # Check cache first
    if use_cache and _split_data_cache is not None:
        return _split_data_cache
    
    # Load and preprocess
    df = load_data(use_cache=use_cache)
    X, y, encoders = preprocess_data(df, save_encoders=False)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(...)
    
    # Encode target
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    # Cache result
    _split_data_cache = (X_train, X_test, y_train, y_test, label_encoder, encoders)
    
    return X_train, X_test, y_train, y_test, label_encoder, encoders
```

## 📋 Feature Information

### **Total Features**: 58 (sau encoding)

**Feature Categories**:
1. **Demographic**: Age, BMI, Ethnicity, Socioeconomic Factors
2. **Clinical**: Blood Pressure, Cholesterol, Blood Glucose, Waist Circumference
3. **Genetic**: Genetic Markers, Autoantibodies, Family History
4. **Lifestyle**: Physical Activity, Dietary Habits, Smoking, Alcohol
5. **Medical History**: PCOS, Gestational Diabetes, Pregnancy History
6. **Test Results**: Glucose Tolerance Test, Liver Function, Urine Test
7. **Condition-Specific**: 
   - Cystic Fibrosis Diagnosis
   - Pancreatic Health
   - Pulmonary Function
   - Steroid Use History
   - Neurological Assessments

## 🔄 Data Flow trong Training

```
1. get_prepared_data() được gọi lần đầu
   ↓
2. Load processed CSV (cached)
   ↓
3. Preprocess (nếu chưa cache)
   - Encode categoricals
   - Encode target
   ↓
4. Split train/test (stratified)
   ↓
5. Cache kết quả
   ↓
6. Các model sau sử dụng cache
   ↓
7. Mỗi model apply preprocessing riêng:
   - KNN/Naive Bayes/Logistic: StandardScaler
   - RF/XGBoost/LightGBM: Direct usage
```

## 🎯 Data Validation

**Trong preprocessing**:
- ✅ Check missing values
- ✅ Check duplicate rows
- ✅ Validate target distribution
- ✅ Check feature types
- ✅ Validate data ranges

**Trong training**:
- ✅ Check empty folds
- ✅ Validate predictions
- ✅ Check score validity (NaN/inf)

## 💾 Model-Specific Encoders

Mỗi model lưu riêng encoders để đảm bảo:
- **Consistency**: Encoding giống hệt training
- **Compatibility**: Có thể load model độc lập
- **Flexibility**: Mỗi model có thể có preprocessing khác

**Saved Files**:
- `{model_name}_encoders.pkl` - Feature encoders
- `{model_name}_label_encoder.pkl` - Target encoder
- `{model_name}_model.pkl` - Trained model

## 🔍 Data Statistics

- **Total Samples**: 70,000
- **Train Samples**: 56,000 (80%)
- **Test Samples**: 14,000 (20%)
- **Features**: 58 (numerical + encoded categorical)
- **Target Classes**: 13
- **Class Distribution**: Imbalanced (cần class_weight='balanced' cho một số models)

