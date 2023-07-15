import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('iris.csv')
print(df)

X= df.iloc[:,1:5]
y=df.iloc[:,5]

print(X)
print(y)
# train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

# train_the_model

model = RandomForestClassifier()
model.fit(X_train,y_train)

print('[INFO] model is trained')

accuracy = model.score(X_test,y_test)
print(f'[INFO] the accuracy of model is {accuracy}')

import joblib
joblib.dump(model,'model_saved.pkl')
print('[INFO] saving is done')


