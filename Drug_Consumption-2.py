#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings


# In[2]:


#important libraries and main style
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


# In[3]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


# In[5]:


#read the csv file & making a copy of the df

df = pd.read_csv('/Users/GJR/Documents/Drug test/drug_consumption.csv')
copy_df = df.copy() #i made a copy of the dataframe here


# In[6]:


df.info()


# In[7]:


df.head()


# In[8]:


df.describe()


# In[10]:


#check for null val
df.isna().sum()


# In[11]:


originalFeatures = df
print ('originalFeatures count',len(originalFeatures))
print('originalFeatures',originalFeatures)
df.head()


# In[12]:


feature_col_names = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
plt.style.use('ggplot')
f, ax = plt.subplots(figsize=(11, 15))
ax.set(xlim=(-.100, 5))
plt.ylabel('Dependent Variable')
plt.title('Box Plot of Pre-Processed Data Set')
ax = sns.boxplot(data = df[feature_col_names],
  orient = 'h',
  palette = 'Set2')


# In[13]:


columns = ['Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
           'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'Semer', 'VSA']
cp = ['User_Alcohol','User_Amphet', 'User_Amyl', 'User_Benzos', 'User_Caff', 'User_Cannabis', 'User_Choc', 'User_Coke', 'User_Crack',
           'User_Ecstasy', 'User_Heroin', 'User_Ketamine', 'User_Legalh', 'User_LSD', 'User_Meth', 'User_Mushrooms','User_Nicotine', 'User_Semer', 'User_VSA']


# In[14]:


from sklearn.preprocessing import LabelEncoder
for column in columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


# In[15]:


for column in columns:
    le = LabelEncoder()
    copy_df[column] = le.fit_transform(df[column])


# In[16]:


for column in columns:
    le = LabelEncoder()
    copy_df[column] = le.fit_transform(copy_df[column])


# In[17]:


#new column for each drug which contain info of use or not

for i in range(len(columns)):
    copy_df.loc[(copy_df[columns [i]]==0 | (copy_df [columns [i]]==1)), cp[i]] = 'Non-user'
    copy_df.loc[((copy_df[columns[i]]==2) | (copy_df[columns[i]]==3) | (copy_df[columns[i]]==4) | (copy_df[columns[i]]==5) | (copy_df[columns[i]]==6)),cp[i]] = 'User'


# In[18]:


fig, axes = plt.subplots(5,3,figsize = (16,16))
fig.suptitle("Count of Different Classes Vs Drug",fontsize=14)
k=0
for i in range(5):
    for j in range(3):
        sns.countplot(x=columns[k], data=copy_df,ax=axes[i][j])
        k+=1
        
plt.tight_layout()
plt.show()


# In[19]:


count_of_users = []
count_of_non_users = []


# In[20]:


for i in range(len(columns)):
    s = copy_df.groupby([cp[i]]) [columns[i]].count()
    count_of_users.append(s[1])
    count_of_non_users.append(s[0])


# In[21]:


trace1 = go.Bar(
    x=columns,
    y=count_of_users,
    name='User',
    marker = dict(color="rgb(117, 127, 221)"))

trace2 = go.Bar(
    x=columns,
    y=count_of_non_users,
    name='Non-User',
    marker = dict(color="rgb(191, 221, 229)"))


data = [trace1, trace2]
layout = go.Layout(
    title= 'Drug Vs User Or Non-user',
    yaxis=dict(title='Count', ticklen=5, gridwidth=2),
    barmode='group')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[22]:


bins =  np.arange(1,20,1)
plt.figure(figsize=(16,6))
plt.bar(bins+0, count_of_users,width=0.4, label ='User')
plt.bar(bins+.30,count_of_non_users,width=0.4,label ='Non-User')
plt.xticks(bins, columns, rotation=50, fontsize=13)
plt.ylabel("Count",fontsize=13)
plt.title("Drug Vs User or Non-user",fontsize=15)
plt.legend()


# In[23]:


ax = sns.countplot(x='Age', data=df)
plt.title('Age Vs Count')
ax.figure.set_size_inches(8,5)


# In[28]:


for column in copy_df[columns]:
    le = LabelEncoder()
    copy_df[column] = le.fit_transform(copy_df[column])


# In[29]:


for column in df[columns]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


# In[30]:


copy_df['Count'] = pd.Series()
copy_df['Count'] = copy_df['Count'].fillna(value = 0.0)
for i in cp:
    copy_df['Count']+=copy_df[i]


# In[31]:


pk = copy_df['Count'].value_counts()


# In[32]:


col = [i for i in range(len(pk.values))]
data = [
go.Bar(
    x = list(pk.index),
    y = list(pk.values),
    marker=dict(color=col, colorscale='Jet', showscale=False)
),]
layout= go.Layout(
    title= 'Used Drugs Vs Number of Users',
    yaxis=dict(title='Count', ticklen=5, gridwidth=2),
    xaxis=dict(title='Drug Count', ticklen=5, gridwidth=2),
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Drug-Count')


# In[33]:


df['Country'].value_counts()
con = ['UK','USA','Canada','Australia','Ireland','New Zealand']


# In[34]:


data = [dict(
        type='choropleth',
        locations = con,
        locationmode='country names',
        z=(df['Country'].value_counts().values),
        text=con,
        colorscale='prtland',
        reversescale=True,
)]
layout = dict(
    title = 'A Map About The Population of Drug Addicted in Each Country',
    geo = dict(showframe=False, showcoastline=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='world-map')


# In[35]:


plt.figure(figsize=(12,15))
sns.violinplot(x='Age', y='Impulsive', data=df)
plt.title('Violin plot of Age by Impulsive',fontsize=14)
plt.xlabel('Impulsive',fontsize=13)
plt.ylabel('Age', fontsize=13)
plt.show()


# In[36]:


plt.figure(figsize=(12,5))
sns.violinplot(x='Age', y='Impulsive', data=df)
plt.title('Violin plot of Age by Impulsive',fontsize=14)
plt.xlabel('Impulsive',fontsize=13)
plt.ylabel('Age',fontsize=13)
plt.show()


# In[37]:


corrmat = df.corr()
plt.figure(figsize=(20,20))
sns.set(font_scale=1)
hm = sns.heatmap(corrmat,cmap = 'RdYlGn',annot=True,
yticklabels = df.columns, xticklabels = df.columns)
plt.xticks(fontsize=13, rotation=50)
plt.yticks(fontsize=13)
plt.title('Correlation B/W Different Features', fontsize=18)
plt.show()


# In[38]:


yp = []
for i in df['Benzos']:
    if(i==0):
        yp.append([1,0,0,0,0,0,0])
    elif(i==1):
        yp.append([0,1,0,0,0,0,0])
    elif(i==2):
        yp.append([0,0,1,0,0,0,0])
    elif(i==3):
        yp.append([0,0,0,1,0,0,0])
    elif(i==4):
        yp.append([0,0,0,0,1,0,0])
    elif(i==5):
        yp.append([0,0,0,0,0,1,0])
    elif(i==6):
        yp.append([0,0,0,0,0,0,1])
yp = np.array(yp)


# In[39]:


from sklearn.model_selection import train_test_split
feature_col_names = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
predicted_class_names = ['Benzos']

X = df[feature_col_names].values
y = df[predicted_class_names].values

X_train, X_test, y_train, y_test = train_test_split(X, yp, test_size=0.30, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, yp, test_size=0.30, random_state=42)


# In[40]:


#backpropagation

num_inputs = len(X_train[0])
hidden_layer_neurons = 13
np.random.seed(4)
b1 = 2*np.random.random(num_inputs) -1
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
w1


# In[41]:


num_outputs = 7
b2 = 2*np.random.random(num_inputs) -1
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs))-1
w2


# In[42]:


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)


# In[43]:


fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [12, 13, 7])


# In[44]:


# sigmoid fx rep
xp = np.linspace(-5, 5, 50)
yp = 1/(1+np.exp( -xp))
plt.plot( xp,yp)


# In[45]:


error = []
b1=0
b2=0
learning_rate = 0.2

#this gradually update the network

for epoch in range (1000):
    l1 = 1/(1 + np.exp(-(np.dot(X_train, w1)+b1 )))
    
    #sigmoid fx
    
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2) +b2 )))
    er = (abs(y_train - l2)).mean()
    l2_delta = (y_train - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T)*(l1*(1-l1))
    w2 += l1.T.dot(l2_delta)*learning_rate
    
    w1 += X_train.T.dot(l1_delta) * learning_rate
    error.append(er/(epoch*0.1))
    print('Error:', er)
          


# In[46]:


er = set(error)


# In[47]:


sp = pd.Series(error)
sp.plot()
plt.title('Epoch Vs Error Rate', fontsize=13)
plt.xlabel('Epoch')


# In[48]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[49]:


import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing.text import Tokenizer


# In[50]:


# Adding the input layer and the first hidden layer
#rectified linear unit
#classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
# Adding the second hidden layer
# Adding dropout to prevent overfitting
# Adding the output layer


# In[51]:


classifier = Sequential()
    classifier.Dense(16, activation='relu', input_dim = 12),


    classifier.Dense(16, activation='relu'),



    classifier.Dense(7, activation='sigmoid'),


# In[52]:


classifier = Sequential()
# Adding the input layer and the first hidden layer
#rectified linear unit
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=12))
#classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))

# Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(output_dim=7, init='uniform', activation='sigmoid'))


# In[53]:


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[54]:


classifier.fit(X_train, y_train, validation_split = 0.20, batch_size=100, epochs=70, verbose=1)


# In[55]:


classifier.fit(X_train, y_train, validation_split = 0.20, batch_size=100, epochs=70,verbose=1)


# In[56]:


# Predicting the Test set results
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = classifier.predict(X_test)
y_pred


# In[57]:


for i in range(len(y_pred)):
    maxs = max(y_pred[i])
    #print(maxs)
    y_pred[i] = (y_pred[i]==maxs)


# In[58]:


#print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
#accuracy_score(y_test, y_pred)


# In[59]:


from sklearn import metrics
def plot_confusion_metrix(y_test,model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-User','User']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[60]:


from sklearn.svm import SVC
svm = SVC(kernel="rbf", C=2,random_state=0)
svm.fit(X_train1, y_train1.ravel())


# In[61]:


pred = svm.predict(X_test)
accu = metrics.accuracy_score(y_test1,pred)
accu


# In[62]:


feature_col_names = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
predicted_class_names = ['User_Benzos']

X = copy_df[feature_col_names].values
y = copy_df[predicted_class_names].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[63]:


from sklearn.metrics import roc_curve,auc

def report_performance(model):

    model_test = model.predict(X_test)

    print("\n\nConfusion Matrix:")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("\n\nClassification Report: ")
    print(metrics.classification_report(y_test, model_test))
    #cm = metrics.confusion_matrix(y_test, model_test)
    plot_confusion_metrix(y_test, model_test)


def roc_curves(model):
    predictions_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(predictions_test,y_test)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def accuracy(model):
    pred = model.predict(X_test1)
    accu = metrics.accuracy_score(y_test,pred)
    print("\nAcuuracy Of the Model: ",accu,"\n\n")
    #total_accuracy[str((str(model).split('(')[0]))] = accu


# In[64]:


svm = SVC(kernel="linear", C=1,random_state=0)
svm.fit(X_train, y_train.ravel())
report_performance(svm) 
roc_curves(svm)


# In[65]:


from sklearn.tree import DecisionTreeClassifier
clf_dtc = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=0)
clf_dtc.fit(X_train, y_train.ravel())
report_performance(clf_dtc) 
roc_curves(clf_dtc)
pred = svm.predict(X_test)
accu = metrics.accuracy_score(y_test1,pred)


# In[66]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train,y_train)


# In[67]:


report_performance(clf)
roc_curves(clf)


# In[68]:


feature_col_names = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
predicted_class_names = ['Benzos']

X = df[feature_col_names].values
y = df[predicted_class_names].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.30, random_state=42)


# In[69]:


clf = KNeighborsClassifier()
clf.fit(X_train1,y_train1)


# In[70]:


pred = clf.predict(X_test1)
#print(pred)
accu = metrics.accuracy_score(y_test1,pred)


# In[71]:


from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='dcm.html')


# In[ ]:




