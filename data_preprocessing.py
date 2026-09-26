import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load Data
df = pd.read_csv("your_dataset.csv")

# 2. Exploratory Data Analysis (EDA)
print("--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe(include="all"))
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Separate features (X) and target variable (y)
# Adjust 'target_column' to match your actual dataset
X = df.drop(columns=["target_column"])
y = df["target_column"]

# Identify Numerical and Categorical Columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 3. Handle Missing Values
# Numerical: Impute with median
num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Categorical: Impute with most frequent value (mode)
cat_imputer = SimpleImputer(strategy="most_frequent")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# 4. Encode Categorical Variables
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_cats = encoder.fit_transform(X[cat_cols])
encoded_cat_names = encoder.get_feature_names_out(cat_cols)
encoded_cat_df = pd.DataFrame(
    encoded_cats, columns=encoded_cat_names, index=X.index
)

# Combine numerical features and encoded categorical features
X_processed = pd.concat([X[num_cols], encoded_cat_df], axis=1)

# 5. Feature Scaling (Normalization/Standardization)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_processed), columns=X_processed.columns
)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Export cleaned data for validation
X_scaled.to_csv("cleaned_features.csv", index=False)
print("\nPreprocessing complete. Saved 'cleaned_features.csv'.")
