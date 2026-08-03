import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD THE NEW KAGGLE DATASET
cars = pd.read_csv("cardekho.csv")
print("Initial Data details")
print(f"Original Data Shape : {cars.shape}")

# Remove explicit null rows
cars = cars.dropna()

# 2. EXTRACT THE BRAND NAME FROM THE 'name' COLUMN
cars['brand'] = cars['name'].str.split(' ').str[0]

# 3. CLEAN UP TEXT UNITS FROM NUMERIC COLUMNS SAFELY
# Convert to string first to ensure .str operations do not crash on numeric types
if 'mileage(km/ltr/kg)' in cars.columns:
    cars['mileage(km/ltr/kg)'] = cars['mileage(km/ltr/kg)'].astype(str).str.split(' ').str[0]

if 'engine' in cars.columns:
    cars['engine'] = cars['engine'].astype(str).str.split(' ').str[0]

if 'max_power' in cars.columns:
    cars['max_power'] = cars['max_power'].astype(str).str.split(' ').str[0]
    # Replace empty string spaces ' ' with NaN so they can be dropped properly
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
# filter out columns that don't exist just in case
categorical_cols = [col for col in categorical_cols if col in cars.columns]
cars = pd.get_dummies(
    cars, columns=categorical_cols, drop_first=True, dtype=int
)

# 5. CREATE TARGET PRICE TIERS (Low, Medium, High)
cars["Price_Tier"] = pd.qcut(
    cars["selling_price"], q=3, labels=["Low", "Medium", "High"]
)

# Convert Price_Tier to numbers temporarily to calculate brand strength
tier_mapping = {"Low": 1, "Medium": 2, "High": 3}
cars["Tier_Num"] = cars["Price_Tier"].map(tier_mapping).astype(int)

# Calculate the average tier for each brand
brand_means = cars.groupby("brand")["Tier_Num"].mean()

# Replace the text "brand" column with these numerical averages
cars["Brand_Encoded"] = cars["brand"].map(brand_means)

# Drop the temporary column
cars = cars.drop(columns=["Tier_Num"])

# 6. SEPARATE FEATURES (X) FROM THE TARGETS (y)
columns_to_drop = ["name", "brand", "owner", "selling_price", "Price_Tier"]
X = cars.drop(columns=columns_to_drop, errors="ignore")

y_class = cars["Price_Tier"]  # Classification target (For KNN, SVM, ANN)

# Split into training and testing data (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# Normalize features (Scale between 0 and 1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. VERIFY PREPROCESSING SUCCESS
# ==========================================
print("\n--- DATA PREPROCESSING COMPLETE ---")
print(f"Features trained shape: {X_train_scaled.shape}")
print(f"Features tested shape : {X_test_scaled.shape}")
print("\nAll categorical features converted and values normalized successfully!")

# Print original columns vs columns after encoding/dropping
print("\n--- COLUMN SELECTION CHANGES ---")
print(f"Original Columns ({len(cars.columns)}): \n{list(cars.columns)}")

# Temporary DataFrame to see columns left in X before converting to NumPy
print(f"\nProcessed Feature Columns ({len(X.columns)}): \n{list(X.columns)}")

print("\n--- FINAL MODEL READINESS CHECK ---")
print(f"Any missing values in X_train: {np.isnan(X_train_scaled).any()}")
print(f"Any missing values in X_test : {np.isnan(X_test_scaled).any()}")
print("-----------------------------------")

# test dataset export
test_data_export = X_test.copy()
test_data_export['Price_Tier'] = y_test
test_data_export.to_csv("test_dataset_20percent.csv", index=False)

print("Saved 20% test dataset to 'test_dataset_20percent.csv'!")