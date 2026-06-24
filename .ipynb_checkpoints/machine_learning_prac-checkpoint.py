# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
X_new = np.array([

    [10, 2],

    [50, 5]

])


churn_df = pd.read_csv("telecom_churn.csv")

y = churn_df["Churn"].values
X = churn_df[["tenure", "MonthlyCharges"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)
#Now that your KNN classifier has been fit to the data,
#it can be used to predict the labels of new data points.

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred))

#The model has predicted the first and third customers
#will not churn in the new array
