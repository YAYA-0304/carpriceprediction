import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD THE DATASET
cars = pd.read_csv("cardekho.csv")
cars = cars.dropna().drop_duplicates()

# 2. EXTRACT BRAND NAME
cars['brand'] = cars['name'].str.split(' ').str[0]

# 3. CLEAN UP TEXT UNITS
if 'mileage(km/ltr/kg)' in cars.columns:
    cars['mileage(km/ltr/kg)'] = cars['mileage(km/ltr/kg)'].astype(str).str.split(' ').str[0]

if 'engine' in cars.columns:
    cars['engine'] = cars['engine'].astype(str).str.split(' ').str[0]

if 'max_power' in cars.columns:
    cars['max_power'] = cars['max_power'].astype(str).str.split(' ').str[0]
    cars['max_power'] = cars['max_power'].replace(r'^\s*$', np.nan, regex=True)

for col in ['mileage(km/ltr/kg)', 'engine', 'max_power']:
    if col in cars.columns:
        cars[col] = pd.to_numeric(cars[col], errors='coerce')

cars = cars.dropna()

# 4. ONE-HOT ENCODING (Categorical columns)
categorical_cols = ["fuel", "transmission", "seller_type"]
categorical_cols = [col for col in categorical_cols if col in cars.columns]
cars = pd.get_dummies(cars, columns=categorical_cols, drop_first=True, dtype=int)

# 5. SEPARATE FEATURES AND CONTINUOUS TARGET BEFORE TIER CREATION
columns_to_drop = ["name", "owner", "selling_price"]
X = cars.drop(columns=columns_to_drop, errors="ignore")
y_price = cars["selling_price"]

# Initial Split (80/20)
X_train, X_test, y_train_price, y_test_price = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

# 6. CREATE PRICE TIERS STRICTLY ON TRAINING BINS 
_, bin_edges = pd.qcut(y_train_price, q=3, retbins=True, labels=["Low", "Medium", "High"])
bin_edges[0] = -np.inf  # Ensure min edge covers any lower test prices
bin_edges[-1] = np.inf  # Ensure max edge covers any higher test prices

y_train = pd.cut(y_train_price, bins=bin_edges, labels=["Low", "Medium", "High"])
y_test = pd.cut(y_test_price, bins=bin_edges, labels=["Low", "Medium", "High"])

# 7. TARGET ENCODE BRAND ON TRAINING LABELS ONLY
tier_mapping = {"Low": 1, "Medium": 2, "High": 3}
y_train_num = y_train.map(tier_mapping).astype(int)

brand_means = y_train_num.groupby(X_train["brand"]).mean()
global_mean = y_train_num.mean()

X_train["Brand_Encoded"] = X_train["brand"].map(brand_means).fillna(global_mean)
X_test["Brand_Encoded"] = X_test["brand"].map(brand_means).fillna(global_mean)

# Drop raw 'brand' column
X_train = X_train.drop(columns=["brand"])
X_test = X_test.drop(columns=["brand"])

# 8. NORMALIZE FEATURES (Fit on Train only)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. SAVE TEST DATASET
raw_cars = pd.read_csv("cardekho.csv")
test_df = raw_cars.loc[X_test.index].copy()
test_df['Price_Tier'] = y_test
test_df.to_csv("test_dataset.csv", index=False)

print("Data Preprocessing & Zero-Leakage Pipeline Complete!")