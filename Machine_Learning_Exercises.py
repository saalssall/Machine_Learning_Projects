import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Optional debug (you can remove later)
print(sys.executable)
print(sys.version)

# Load dataset
churn_df = pd.read_csv("telecom_churn.csv")

# Features and target (NOTE: adjust column names if needed)
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=6)

# Train model
knn.fit(X, y)

# New data for prediction (MUST be defined)
X_new = [
    [10, 2],
    [50, 5]
]

# Predict
y_pred = knn.predict(X_new)

print("Predictions:", y_pred)
