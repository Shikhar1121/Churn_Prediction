# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 06:41:18 2020

@author: Shikhar
"""
#Part 1 Data Preprocessing
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Churn_Modelling.csv")
x = df.iloc[:,3:13].values
y = df.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
lb_1 = LabelEncoder()
lb_2 = LabelEncoder()
x[:,1] = lb_1.fit_transform(x[:,1])
x[:,2] = lb_2.fit_transform(x[:,2])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)
x = x[:,1:]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2 ,random_state = 56)

from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
sc= StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)




classifier = Sequential()

classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer ='uniform',input_dim = 11))
classifier.add(Dropout(rate =0.1))

classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer ='uniform'))
classifier.add(Dropout(rate =0.1))


classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer= 'uniform'))


classifier.compile(optimizer='adam', loss = 'binary_crossentropy',metrics= ['accuracy'])




classifier.fit(train_x,train_y,batch_size =25,epochs = 500)

y_pred = classifier.predict(test_x)
y_pred = (y_pred > 0.5)


new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y,y_pred )


import keras
from keras.models import Sequential
from keras.layers import Dense


from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score



def classifier_build():
    classifier = Sequential()

    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer ='uniform',input_dim = 11))

    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer ='uniform'))

    classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer= 'uniform'))


    classifier.compile(optimizer='adam', loss = 'binary_crossentropy',metrics= ['accuracy'])
    return classifier    
classifier = KerasClassifier(build_fn=classifier_build,batch_size =10,epochs = 100)
acc = cross_val_score(estimator = classifier,X = train_x,y= train_y,cv= 10,n_jobs = -1)
mn = acc.mean()
var = acc.std()

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



def classifier_build(optimizer):
    classifier = Sequential()

    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer ='uniform',input_dim = 11))

    classifier.add(Dense(units = 6,activation= 'relu',kernel_initializer ='uniform'))

    classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer= 'uniform'))


    classifier.compile(optimizer=optimizer, loss = 'binary_crossentropy',metrics= ['accuracy'])
    return classifier    
classifier = KerasClassifier(build_fn=classifier_build)


para = {'batch_size':[25,32],'epochs': [100,500],
        'optimizer':['adam','rmsprop']}

gscv = GridSearchCV(estimator = classifier,
                    param_grid= para,
                    scoring ='accuracy',
                    cv = 10)

gscv = gscv.fit(train_x,train_y)

best_para =gscv.best_params_ 
best_acc = gscv.best_score_

classifier.fit(train_x,train_y,batch_size =25,epochs = 500)

y_pred = classifier.predict(test_x)
y_pred = (y_pred > 0.5)


