# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the train and test dataset/files
train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')
#train['TARGET'].value_counts()
#print(train.head())
#print(train.dtypes)

#Function to print categorical and numeric features
#def type_features(data):
#    categorical_features = data.select_dtypes(include = ["object"]).columns#include all the columns with data type as object
#    numerical_features = data.select_dtypes(exclude = ["object"]).columns # include columns with data type other than object
#    print( "categorical_features :",categorical_features)
#    print('-----'*40)
#    print("numerical_features:",numerical_features)
#type_features(train)
#type_features(test)

#Taking care of categorical columns
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
le = LabelEncoder()
le_count = 0
# Iterate through the columns
for col in train:
    if train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train[col].unique())) <= 2:
            # Train on the training data
            le.fit(train[col])
            # Transform both training and testing data
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

#print('Training Features shape: ', train.shape)
#print('Testing Features shape: ', test.shape)

#Identifying missing data
#def missingdata(data):
#    total = data.isnull().sum().sort_values(ascending = False)# Calculate total number of nan values
#    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)#Calculate percentage of nan values out of total
#    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Creating the table
#    ms= ms[ms["Percent"] > 0]
#    f,ax =plt.subplots(figsize=(15,10))
#    plt.xticks(rotation='90')
#    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
#    plt.xlabel('Features', fontsize=15)
#    plt.ylabel('Percent of missing values', fontsize=15)
#    plt.title('Percent missing data by feature', fontsize=15)
#    #ms= ms[ms["Percent"] > 0]
#    return ms
#missingdata(train)
#missingdata(test)

#imputing missing values with a mean
train=train.fillna(train.mean())
test=test.fillna(test.mean())


# Align the training and testing data, keep only columns present in both dataframes
train_labels = train['TARGET']
train, test = train.align(test, join = 'inner', axis = 1)
# Add the target back in
train['TARGET'] = train_labels
#print('Training Features shape: ', train.shape)
#print('Testing Features shape: ', test.shape)

#Splitting train dataset, training the model
y=train.iloc[:,train.columns.get_loc('TARGET')].values
y.reshape(-1,1)
y=y.ravel()
X=train.drop(['TARGET'], axis=1).values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Imputer to replace NANs
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
train = imputer.fit_transform(train)
test = imputer.fit_transform(test)

## Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifierLR = LogisticRegression(random_state = 0)
#classifierLR.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = classifierLR.predict(X_test)
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cmLR = confusion_matrix(y_test, y_pred)
#print(cmLR)
#
#log_reg_pred = classifierLR.predict_proba(test)[:, 1]
# Fitting Logistic Regression to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifierKNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmKNN = confusion_matrix(y_test, y_pred)
print(cmKNN)

log_reg_pred = classifierKNN.predict_proba(test)[:, 1]
# Submission dataframe
submit = test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
submit.to_csv('log_reg_baseline4.csv', index = False)