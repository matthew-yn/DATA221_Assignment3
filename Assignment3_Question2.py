#Imports pandas package as pd and matplotlib package as plt.
import pandas as pd
import matplotlib.pyplot as plt

#Reads CSV file.
crime = pd.read_csv("crime1.csv")
#Finds ViolentCrimesPerPop column.
violent_crimes = crime["ViolentCrimesPerPop"]

#Generates histogram.
plt.figure()
plt.hist(violent_crimes)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")
plt.show()

#Generates boxplot.
plt.figure()
plt.boxplot(violent_crimes)
plt.title("Boxplot of Violent Crimes Per Population")
plt.xlabel("ViolentCrimesPerPop")
plt.show()

'''
The histograms shows how the violent crime rates is distributed across a population
by looking at the height of the bars as well as the general distribution of the data.

The box plot shows the median as a highlighted line inside the boxplot. It also shows 
how the data is spread around the median, whether it is spread out wide or clustered 
tightly.

If points are plotted outside the box it represents potential outliers in the data.
'''