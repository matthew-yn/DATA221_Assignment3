#Imports pandas package as pd.
import pandas as pd

#Reads CSV file.
crime = pd.read_csv("crime1.csv")
#Finds ViolentCrimesPerPop column.
violent_crimes = crime["ViolentCrimesPerPop"]

#Computes mean, median, standard deviation, min, and max of column.
mean_crimes = violent_crimes.mean()
median_crimes = violent_crimes.median()
std_crimes = violent_crimes.std()
min_crimes = violent_crimes.min()
max_crimes = violent_crimes.max()

#Prints computed values to user.
print("Mean:", mean_crimes)
print("Median:", median_crimes)
print("Standard Deviation:", std_crimes)
print("Min:", min_crimes)
print("Max:", max_crimes)

'''
Since the mean is greater than the median, the distribution is right-skewed.

If there are extreme values in the data set, it would affect the mean more than
the median as the mean computes all values in the data set while median only takes
the middle value.
'''