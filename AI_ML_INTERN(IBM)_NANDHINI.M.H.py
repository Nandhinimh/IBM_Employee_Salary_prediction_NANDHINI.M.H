#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv(r"C:\Users\ADMIN\Desktop\data01.csv")
data


# In[3]:


data.workclass.replace({'?':'Notlisted'},inplace=True)
print(data.workclass.value_counts())


# In[4]:


#remove unnecessary attributes in the features
data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']
print(data.workclass.value_counts())


# In[5]:


data=data[data['education']!='5th-6th']
data=data[data['education']!='1st-4th']
data=data[data['education']!='Preschool']
print(data.education.value_counts())


# In[6]:


#to remove unwanted features
data.drop(columns=['education'],inplace=True)
data.head(5)


# In[7]:


#outlier
plt.boxplot(data['age'])
plt.show()
          


# In[8]:


#data w/o outliers
data=data[(data['age']<=75 )& (data['age']>=17)]


# In[9]:


plt.boxplot(data['age'])
plt.show()


# In[10]:


#label encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['native-country']=encoder.fit_transform(data['native-country'])
data


# In[11]:


x=data.drop(columns=['income'])
y=data['income']
x


# In[12]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x


# In[13]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler',StandardScaler()),
        ('model',model)
    ])

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# In[14]:


import matplotlib.pyplot as plt

plt.bar(results.keys(), results.values(),color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[25]:


import joblib

#Train-test split
X_train,X_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Define models
models = { 
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}
results={}
#Train and evaluate
for name, model in models.items():
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    acc=accuracy_score(y_test,preds)
    results[name]=acc
    print(f"{name}:{acc:.4f}")

#Get best model
best_model_name = max(results,key=results.get)
best_model = models[best_model_name]
print(f"\n Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

#Save the best model
joblib.dump(best_model,"best_model.pkl")
print("Saved best model as best_model.pkl")

