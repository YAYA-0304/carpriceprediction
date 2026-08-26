import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD THE NEW KAGGLE DATASET
cars = pd.read_csv("cardekho.csv")
print("Initial Data details")
print(f"Original Data Shape : {cars.shape}")

# Check for exact duplicate rows
duplicate_count = cars.duplicated().sum()
print(f"Total duplicate rows in dataset: {duplicate_count}")

# Remove explicit null rows
cars = cars.dropna().drop_duplicates()

# 2. EXTRACT THE BRAND NAME FROM THE 'name' COLUMN
cars['brand'] = cars['name'].str.split(' ').str[0]

# 3. CLEAN UP TEXT UNITS FROM NUMERIC COLUMNS SAFELY
if 'mileage(km/ltr/kg)' in cars.columns:
    cars['mileage(km/ltr/kg)'] = cars['mileage(km/ltr/kg)'].astype(str).str.split(' ').str[0]

if 'engine' in cars.columns:
    cars['engine'] = cars['engine'].astype(str).str.split(' ').str[0]

if 'max_power' in cars.columns:
    cars['max_power'] = cars['max_power'].astype(str).str.split(' ').str[0]
    cars['max_power'] = cars['max_power'].replace(r'^\s*$', np.nan, regex=True)

# Convert these cleaned columns to actual floating-point numbers
numeric_cols_to_convert = ['mileage(km/ltr/kg)', 'engine', 'max_power']
for col in numeric_cols_to_convert:
    if col in cars.columns:
        cars[col] = pd.to_numeric(cars[col], errors='coerce')

# Drop rows that ended up with NaN values after converting empty text strings
cars = cars.dropna()

# 4. CONVERT CATEGORICAL DATA INTO NUMBERS (One-Hot Encoding)
categorical_cols = ["fuel", "transmission", "seller_type"]
categorical_cols = [col for col in categorical_cols if col in cars.columns]
cars = pd.get_dummies(
    cars, columns=categorical_cols, drop_first=True, dtype=int
)

# 5. CREATE TARGET PRICE TIERS (Low, Medium, High)
cars["Price_Tier"] = pd.qcut(
    cars["selling_price"], q=3, labels=["Low", "Medium", "High"]
)

# 6. SEPARATE FEATURES (X) FROM TARGETS (y_class & y_price)
columns_to_drop = ["name", "owner", "selling_price", "Price_Tier"]
X = cars.drop(columns=columns_to_drop, errors="ignore")

y_class = cars["Price_Tier"]       # Classification target (Low / Medium / High)
y_price = cars["selling_price"]     # Regression target (Continuous Price)

# Split both targets together to ensure index/row alignment
X_train, X_test, y_train, y_test, y_train_price, y_test_price = train_test_split(
    X, y_class, y_price, test_size=0.2, random_state=42, stratify=y_class
)

# 7. ENCODE BRAND USING ONLY TRAINING LABELS (Prevents Data Leakage)
tier_mapping = {"Low": 1, "Medium": 2, "High": 3}
y_train_num = y_train.map(tier_mapping).astype(int)

# Group strictly on X_train to calculate brand strength
brand_means = y_train_num.groupby(X_train["brand"]).mean()
global_mean = y_train_num.mean()

# Map to train and test
X_train["Brand_Encoded"] = X_train["brand"].map(brand_means).fillna(global_mean)
X_test["Brand_Encoded"] = X_test["brand"].map(brand_means).fillna(global_mean)

# Drop raw 'brand' text column
X_train = X_train.drop(columns=["brand"])
X_test = X_test.drop(columns=["brand"])

# Normalize features (Scale between 0 and 1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. SAVE TEST DATASET FOR VERIFICATION
raw_cars = pd.read_csv("cardekho.csv")
test_df = raw_cars.loc[X_test.index].copy()
test_df['Price_Tier'] = y_test
test_df.to_csv("test_dataset.csv", index=False)

print("\n-----------------------------------")
print("Saved Test Dataset to 'test_dataset.csv'!")
print("===================================")

# 9. VERIFY PREPROCESSING SUCCESS
print("\n--- DATA PREPROCESSING COMPLETE ---")
print(f"Features trained shape: {X_train_scaled.shape}")
print(f"Features tested shape : {X_test_scaled.shape}")
print(f"Classification train targets: {y_train.shape}")
print(f"Regression train targets    : {y_train_price.shape}")
print("\n--- FINAL MODEL READINESS CHECK ---")
print(f"Any missing values in X_train: {np.isnan(X_train_scaled).any()}")
print(f"Any missing values in X_test : {np.isnan(X_test_scaled).any()}")
print("-----------------------------------")