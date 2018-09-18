
# coding: utf-8

# In[1]:


#Logetic regression


# In[2]:


#import Libraries


# In[3]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import pandas as pd


# In[6]:


#Importing data set


# In[9]:


dataset=pd.read_csv('/home/raniladkat/Downloads/Logistic_Regression/Logistic_Regression/Social_Network_Ads.csv')


# In[10]:


dataset


# In[11]:


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# In[12]:


X


# In[13]:


y


# In[14]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[15]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[16]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[43]:


# Predicting the Test set results

#newtestdata=pd.read_csv('/home/raniladkat/Downloads/Logistic_Regression/Logistic_Regression/Social_Network_Ads.csv')
#y_pred=classifier.predict_proba(newtestedata)
y_pred = classifier.predict_proba(X_test)
data1=pd.DataFrame(y_pred)
data1.columns=["Rejected","Approved"]
data1


# In[37]:



data1["BoardedProbability"]=data1.Approved.apply(groupbyprob)
data1


# In[23]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[29]:


##Converting test data into Probability

def groupbyprob(x):
    if x > 0.9:
        return "high"
    elif x > 0.7 and x < 0.09:
        return "medium"
    elif x > 0.5 and x <0.7:
        return "lowmedium"
    else:
        return "Rejected"
    
        
        


# In[19]:


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[20]:


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[39]:


data1.groupby("BoardedProbability").size()


# In[40]:


data1[data1["BoardedProbability"]=="high"]


# In[41]:


from sklearn.metrics import accuracy_score


# In[44]:


y_pred = classifier.predict(X_test)


# In[45]:


y_pred


# In[46]:


accuracy_score(y_test,y_pred)

