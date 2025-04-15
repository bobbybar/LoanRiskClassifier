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

sns.kdeplot(data=df, x="dti", hue="default", fill=True)
plt.title("DTI Distribution by Default")
plt.show() #shows that defaulted loans have a higher DTI

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show() 

#--------------------------------
#data cleaning

#drop unused columns
df = df.drop(columns=["loan_status", "addr_state"])

#check for missing values
df.isnull().sum() #no missing values

#one hot encoding for categorical variables
df = pd.get_dummies(df, columns=[
    "term", "grade", "emp_length", "home_ownership", "purpose"
], drop_first=True)

#split data into features and target
X = df.drop("default", axis=1)
y = df["default"]


#------------------
#modeling
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)







