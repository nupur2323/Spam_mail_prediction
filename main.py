# importing the libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizerfidf means converting the mail data into numarical means ml languages
from sklearn.linear_model import LogisticRegression
# to classify the mail into spam mail
from sklearn.metrics import accuracy_score
# used to evaluate the model accuracy

# data collection and pre-processing
# loading the data from csv file to pandas data fream
raw_mail_data =pd.read_csv('mail_data.csv')
print(raw_mail_data)


# replace null values with null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# printimg first 5 rows
print(mail_data.head())

# cheacking the no of rows and columns
print(mail_data.shape)

# label encoding
# spam mail as 0 , ham mail as 1
mail_data.loc[mail_data['Category'] =='spam','Category',]=0
mail_data.loc[mail_data['Category'] =='ham','Category',]=1

# spam =0 ,ham=1

# saparating the text(x-axis) and label(y-axis)
X=mail_data['Message']
Y=mail_data['Category']
print(X)
print(Y)

# splitting the data into train and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
print(X.shape)
print(X_train.shape)
print(X_test.shape)

#   feacture extraction
# transform the text data to feacture sectors that can be
# used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)
 
 
#  convert Y_test and Y_train values as integer
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
 
print(X_train)

print(X_train_feature)

# training the model
# logistice regrassion

model=LogisticRegression()
# traning the logistic regrassion model with the training data
print(model.fit(X_train_feature,X_train))

# evaluating the trained  model
# prediction on training data
prediction_on_training_data= model.predict(X_train_feature)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy on training data : ',accuracy_on_training_data)


# predictio on the data
prediction_on_test_data=model.predict(X_test_feature)
accuracy_on_test_data= accuracy_score(Y_test,prediction_on_test_data)

print('Accuracy on test data : ',accuracy_on_test_data)
 
 
# building a prediction
input_mail = ["Hello! How's you and how did saturday go? I was just texting to see if you'd decided to do anything tomo. Not that i'm trying to invite myself or anything!"]

# convert text to feacture vectors
input_data_feactures = feature_extraction.transform(input_mail)

# making prediction
prediction = model.predict(input_data_feactures)
print(prediction)

if prediction[0]==1:
    print('Ham mail')
else:
    print('Spam mail')

