#Imports pandas package as pd and train_test_split, KNeighboursClassifier, and required metrics from sklearn.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Reads data from CSV.
kidney_data = pd.read_csv("kidney_disease.csv", na_values = ["?", "\t?"])

#Cleans classification column.
kidney_data["classification"] = kidney_data["classification"].str.strip()
kidney_data["classification"] = kidney_data["classification"].map({
    "ckd": 0,
    "notckd": 1
})

#Finds necessary columns.
data = kidney_data[["age", "bp", "sod", "pot", "hemo", "pcv", "wc", "rc", "classification"]]

#Drops rows with missing values.
data = data.dropna()

#Creates matrix x that excludes classification column.
x = data[["age", "bp", "sod", "pot", "hemo", "pcv", "wc", "rc"]]
#Creats label vector y using classification.
y = data["classification"]

#Splits into 70% training and 30% testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#List of k values to test.
k_values = [1, 3, 5, 7, 9]

#Creates list to store results.
results = []

#Trains each k value.
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_prediction = knn.predict(x_test)
    #Calculates accuracy of k value.
    accuracy = accuracy_score(y_test, y_prediction)
    #Adds k, accuracy tuple to results list.
    results.append((k, accuracy))

#Creates results table.
results_df = pd.DataFrame(results, columns = ["k", "accuracy"])

#Prints results table to user.
print("K Value Accuracy:")
print(results_df)

#Prints most accurate k value to user.
best_k = max(results, key = lambda x: x[1])[0]
print("Best K Value:", best_k)

'''
Changing the k value affects the model by affecting its flexibility.

Small k values can cause overfitting because the model becomes oversensitive to outliers and
extreme values in the training data.

Large k values can cause underfitting because the model becomes undersensitive and may miss
local patterns.
'''