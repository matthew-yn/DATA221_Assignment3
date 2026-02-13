#Imports pandas package as pd and train_test_split from sklearn.
import pandas as pd
from sklearn.model_selection import train_test_split

#Reads data from CSV.
kidney_data = pd.read_csv("kidney_disease.csv")

#Creates matrix x that excludes classification column.
x = kidney_data.drop("classification", axis = 1)
#Creates label vector y using classification column.
y = kidney_data["classification"]

#Splits into 70% training and 30% testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

'''
We should not test on the same data we trained on because the model would memorize
the data instead of learning the general patterns that it contains. This would cause
the model to overperform.

The purpose of the testing set is to allow us to test how well the model performs
on new, unseen data.
'''