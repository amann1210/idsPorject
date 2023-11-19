# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:49:26 2022

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns
import numpy as np
# =============================================================================

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
import itertools
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
# =============================================================================
df = pd.read_excel("data.xlsx")
shape = df.shape
print(df.shape)
#df.info()

#print(df.describe())

#print(df.isnull().sum())

data = df.drop_duplicates(subset ="UNS",)

#dataset splitting
array = df.values
X = array[:,0:5]
Y = array[:,5]
seed = 7
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2 , random_state = seed)

num_folds = 10
num_instances = len(x_train)
seed = 7
scoring = 'accuracy'


counts = Counter(Y)
plt.bar(counts.keys(), counts.values() ,color=['red', 'blue', 'green', 'orange'])
plt.xlabel('Count')
plt.ylabel('UNS')
plt.title('Frequency Distribution of UNS Data')
plt.show()

#plt.figure(figsize=(10,5))
#sns.pairplot(df, hue="UNS")
# =============================================================================
# #histogram
# 
# #STG
# plt.hist(array[:,0:1])
# plt.title('STG')
# plt.show()
# #SCG
# plt.hist(array[:,1:2])
# plt.title('SCG')
# plt.show()
# #STR
# plt.hist(array[:,2:3])
# plt.title('STR')
# plt.show()
# #LPR
# plt.hist(array[:,3:4])
# plt.title('LPR')
# plt.show()
# #PEG
# plt.hist(array[:,4:5])
# plt.title('PEG')
# plt.show()
# =============================================================================


#Data = pd.DataFrame(df)
#print(Data.corr())


#plt.figure(figsize=(15,8))
#sns.heatmap(df.corr(),annot=True,fmt=".0%",cmap = "tab20")
#plt.title('Correlation Matrix',size=25)
#plt.show()




sns.boxplot(data=df, x="STG", y="UNS")
sns.boxplot(data=df, x="SCG", y="UNS")
sns.boxplot(data=df, x="STR", y="UNS")
sns.boxplot(data=df, x="LPR", y="UNS")
sns.boxplot(data=df, x="PEG", y="UNS")

#sns.boxplot(data=df, x="STG")
Q1, Q3 = np.quantile(array[:,0:1], [0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#print(lower_bound)
#print(upper_bound)
#filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]


#logistic regression model
#=======#======================================================================
# =============================================================================
# model1 = LogisticRegression()
# model1.fit(x_train, y_train)
# y_pred = model1.predict(x_test)
# 
# score = accuracy_score(y_test, y_pred)
# print('Accuracy Score = ' + str(score*100))
# 
# cm = confusion_matrix(y_test, y_pred)
# cm_df = pd.DataFrame(cm,index = ['very_low','High','Low','Middle'], 
#                       columns = ['very_low','High','Low','Middle'])
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('Confusion Matrix')
# plt.ylabel('\n \n Actual Values')
# plt.xlabel('\n Predicted Values \n \n Logistic Regression model')
# plt.show()
# 
# =============================================================================
#=============================================================================

#decision tree Classifier

# =============================================================================
# model2 = DecisionTreeClassifier()
# model2.fit(x_train, y_train)
# y_pred = model2.predict(x_test)
# 
# score = accuracy_score(y_test, y_pred)
# print('Accuracy Score = ' + str(score*100))
# 
# cm = confusion_matrix(y_test, y_pred)
# cm_df = pd.DataFrame(cm,index = ['very_low','High','Low','Middle'], 
#                       columns = ['very_low','High','Low','Middle'])
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('Confusion Matrix')
# plt.ylabel('\n \n Actual Values')
# plt.xlabel('\n Predicted Values \n \n Decision Tree classifier')
# plt.show()
# 
# 
# =============================================================================
#Naive Bayes Classifier

# =============================================================================
# model3 = GaussianNB()
# model3.fit(x_train, y_train)
# y_pred = model3.predict(x_test)
# 
# score = accuracy_score(y_test, y_pred)
# print('Accuracy Score = ' + str(score*100))
# 
# cm = confusion_matrix(y_test, y_pred)
# cm_df = pd.DataFrame(cm,index = ['very_low','High','Low','Middle'], 
#                       columns = ['very_low','High','Low','Middle'])
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('Confusion Matrix')
# plt.ylabel('\n \n Actual Values')
# plt.xlabel('\n Predicted Values \n \n Naive Bayes classifier')
# plt.show()
# 
# 
# =============================================================================




#Random Forest
# =============================================================================
# model4 = RandomForestClassifier()
# model4.fit(x_train, y_train)
# y_pred = model4.predict(x_test)
# 
# score = accuracy_score(y_test, y_pred)
# print('Accuracy Score = ' + str(score*100))
# 
# cm = confusion_matrix(y_test, y_pred)
# cm_df = pd.DataFrame(cm,index = ['very_low','High','Low','Middle'], 
#                       columns = ['very_low','High','Low','Middle'])
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('Confusion Matrix')
# plt.ylabel('\n \n Actual Values')
# plt.xlabel('\n Predicted Values \n \n Naive Bayes classifier')
# plt.show()
# 
# 
# =============================================================================


#Support Vector Machine
model5 = SVC()
model5.fit(x_train, y_train)
y_pred = model5.predict(x_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy Score = ' + str(score*100))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,index = ['very_low','High','Low','Middle'], 
                      columns = ['very_low','High','Low','Middle'])
plt.figure(figsize=(7,3))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('\n \n Actual Values')
plt.xlabel('\n Predicted Values \n \n support vector Machine')
plt.show()
