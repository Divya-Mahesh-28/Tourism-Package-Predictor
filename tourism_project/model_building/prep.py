import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop unnecessary columns
df.drop(columns=["CustomerID","Unnamed: 0"], inplace=True)

# Clean 'Gender' column
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Unify 'Unmarried' to 'Single' in 'MaritalStatus'
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# Convert appropriate float columns to int
columns_to_convert_to_int = [
    'Age',
    'NumberOfFollowups',
    'NumberOfTrips',
    'NumberOfChildrenVisiting'
]

for col in columns_to_convert_to_int:
    # Check if the column is float and all values are effectively integers
    if df[col].dtype == 'float64' and (df[col] == df[col].astype(int)).all():
        df[col] = df[col].astype(int)

# Define features (X) and target (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) ratio of ProdTaken consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
