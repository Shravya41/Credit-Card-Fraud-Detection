# Credit-Card-Fraud-Detection
# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Data
data = pd.read_csv("creditcard_data.csv")

# Preprocess Data
data = data.dropna()
data['Time'] = (data['Time'] / 3600).astype(int)  # Convert time to hours

# Feature Engineering
data['AmountRatio'] = data['Amount'] / data['Time']

# Split data
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate Model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
