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

#Trains model.
num_neighbors = 5
knn = KNeighborsClassifier(n_neighbors = num_neighbors)
knn.fit(x_train, y_train)

#Predicts labels of test data.
y_predict = knn.predict(x_test)

#Computes and prints confusion matrix, accuracy, precision, recall, and f1-score to user.
cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:\n", cm)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_predict)
print("Precision:", precision)
recall = recall_score(y_test, y_predict)
print("Recall:", recall)
f1 = f1_score(y_test, y_predict)
print("F1 Score:", f1)

'''
True positive: Model predicts that a patient has kidney disease when they actually do.
True negative: Model predicts that a patient does not have kidney disease when they actually do not.
False positive: Model predicts that a patient has kidney disease when they actually do not.
False negative: Model predicts that a patient does not have kidney disease when they actually do.

Accuracy alone may be misleading as it may make incorrect diagnoses even if the accuracy is seems 
high.

If missing a kidney disease case is very serious, recall (a.k.a. sensitivity) is the most important
metric. This is the most important metric because it ensures the model catches as many true cases 
as possible.
'''