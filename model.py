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

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) # precision, recall are low for our model, now we will try other models

#------------------
#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model with hyperparameters
rf_model = RandomForestClassifier(
    class_weight='balanced',  # Adjusts for class imbalance
    random_state=42,  # Ensures reproducibility
    max_depth=10,  # Limit the depth of trees (prevents overfitting)
    n_estimators=100,  # More trees = more stable model
    min_samples_split=5,  # Prevents creating branches with too few samples
    min_samples_leaf=4)  # Ensures leaf nodes have at least 4 samples

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))

feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns) #still not great, even after hyperparameter tuning
print(feature_importances.sort_values(ascending=False).head(10))

#------------------
# Transforming the data
# Make a copy of the training data so we don't overwrite originals
X_train_log = X_train.copy()
X_test_log = X_test.copy()

# Features to log transform
log_features = ['annual_inc', 'loan_amnt', 'revol_util', 'dti']

# Apply log1p (log(1 + x)) transformation
for col in log_features:
    X_train_log[col] = np.log1p(X_train_log[col])
    X_test_log[col] = np.log1p(X_test_log[col])

# Initialize Random Forest with tuned parameters
rf_model_log = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    max_depth=10,
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=4
)

# Fit the model on the log-transformed data
rf_model_log.fit(X_train_log, y_train)

# Predict on test set
y_pred_log = rf_model_log.predict(X_test_log)

# Evaluate
print(classification_report(y_test, y_pred_log)) # still not great, but better than before. Consider different models or more data
feature_importance_log = pd.Series(
    rf_model_log.feature_importances_, 
    index=X_train_log.columns
).sort_values(ascending=False)

print(feature_importance_log.head(10)) 
# since random forests are tree based models, they do not respond well to feature scaling, hence we don't see much change

#------------------
#SMOTE
from imblearn.over_sampling import SMOTE
# since previous iterations did not yield good results, we will try SMOTE to balance the classes
# Create the SMOTE object
sm = SMOTE(random_state=42)

# Resample training set
X_train_sm, y_train_sm = sm.fit_resample(X_train_log, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_sm.value_counts()) # now we have 628 defaults and 628 no defaults

rf_model_sm = RandomForestClassifier(
    class_weight='balanced',  # still helpful in combination with SMOTE
    random_state=42,
    max_depth=10,
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=4
)

rf_model_sm.fit(X_train_sm, y_train_sm)
y_pred_sm = rf_model_sm.predict(X_test_log)

print(classification_report(y_test, y_pred_sm)) # better precision and recall, but still not great
#------------------
#XGBoost
from xgboost import XGBClassifier






