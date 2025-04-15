import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 


#read data from local csv file
df = pd.read_csv('data/lendingclub_1yr_synthetic.csv')

# Display the first few rows, info, and description of the dataframe
df.head()
df.info()
df.describe()
df['default'].value_counts() #203 defaults, 797 no defaults

sns.boxplot(data=df, x="default", y="loan_amnt")
plt.title("Loan Amount by Default Status")
plt.show() #not much difference in loan amount between default and no default

sns.boxplot(data=df, x="default", y="int_rate")
plt.title("Interest Rate by Default Status")
plt.show() #interest rate is higher for defaulted loans, but not by much




