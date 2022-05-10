#!/usr/bin/env python
# coding: utf-8

# # Supervise Vector Machine
# ### Antonio Rodriguez
# The following notebook shows the step by step done in order to generate a supervise vector machine to analyze a set of papers downloded from the Web Of Science dataset.

# The first thing done is to import useful libreries that will be used in the code

# In[1]:



import pandas as pd #Pandas
import numpy as np #numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import nltk.corpus
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from itertools import chain
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA


# First thing to do is to import the data from the folder and chck length, data type etc...

# In[5]:


# header=0 instead of header=1 (LUISJA)
df_train = pd.read_csv("Data_tra.tsv",header=0, sep='\t') #read the text with pandas
df_evaluation = pd.read_csv("Data_evl.tsv",header=0, sep='\t')
df_development=pd.read_csv("Data_dev.tsv",header=0, sep='\t')


# In[6]:


df_train.head()
#len(df_train) Commented by LUISJA


# In[7]:


df_evaluation.dtypes


# ## Data Processing
# The first step is to process and clean the data. It is important to check which kind of data are we being given and how corrupted it is.
# The data is given new key values so in that way it's easier to call for it.

# In[8]:


# Commmented by LUISJA
#df_train.rename({'0' : 'Y1'},axis='columns',inplace=True)
#df_train.rename({'12' : 'Y2'},axis='columns',inplace=True)
#df_train.rename({'12.1' : 'Y'},axis='columns',inplace=True)
#df_train.rename({'CS' : 'DOMAIN'},axis='columns',inplace=True)
#df_train.rename({'Symbolic computation' : 'area'},axis='columns',inplace=True)
#df_train.rename({df_train.columns[5] : 'keywords'},axis='columns',inplace=True)
#df_train.rename({df_train.columns[6] : 'Abstract'},axis='columns',inplace=True)
df_train.head()


# In[ ]:





# In[9]:


# Commmented by LUISJA
#df_evaluation.rename({'0' : 'Y1'},axis='columns',inplace=True)
#df_evaluation.rename({'12' : 'Y2'},axis='columns',inplace=True)
#df_evaluation.rename({'12.1' : 'Y'},axis='columns',inplace=True)
#df_evaluation.rename({'CS' : 'DOMAIN'},axis='columns',inplace=True)
#df_evaluation.rename({'Symbolic computation' : 'area'},axis='columns',inplace=True)
#df_evaluation.rename({df_evaluation.columns[5] : 'keywords'},axis='columns',inplace=True)
#df_evaluation.rename({df_evaluation.columns[6] : 'Abstract'},axis='columns',inplace=True)
df_evaluation.head()


# In[11]:


df_train['Abstract'][38]


# checking data type

# In[12]:


# Commmented by LUISJA
#df_development.rename({'0' : 'Y1'},axis='columns',inplace=True)
#df_development.rename({'12' : 'Y2'},axis='columns',inplace=True)
#df_development.rename({'12.1' : 'Y'},axis='columns',inplace=True)
#df_development.rename({'CS' : 'DOMAIN'},axis='columns',inplace=True)
#df_development.rename({'Symbolic computation' : 'area'},axis='columns',inplace=True)
#df_development.rename({df_development.columns[5] : 'keywords'},axis='columns',inplace=True)
#df_development.rename({df_development.columns[6] : 'Abstract'},axis='columns',inplace=True)
df_development.head()


# In the cels belowit was used the method .unique() for checking wich kind of data has each column so to check if it fits with the information that we are expecting. The answer is no, because in Y2 it as expected natural numbers but in there is a -1.

# In[13]:


df_train['Y2'].unique()


# In[14]:


df_train['Y'].unique()


# In[15]:


df_train['Domain'].unique() # Domain instead of DOMAIN (LUISJA)


# In[16]:


df_train['area'].unique()


# In[17]:


df_train['keywords'].unique()


# In[18]:


df_evaluation['keywords'].unique()


# ### Next step is to clean the data in order to don't get data that will mess the calculations with the machine

# In[19]:


# Domain instead of DOMAIN (LUISJA)
df_train_no_missing = df_train.loc[(df_train['Y2']!=-1) & (df_train['Y']!=-1) & (df_train['Y1']!=df_train['area'].unique()[1]) & (df_train['Domain']!=df_train['area'].unique()[1]) & (df_train['Abstract']!='-')]
df_development_no_missing = df_development.loc[(df_development['Y2']!=-1) & (df_development['Y']!=-1) & (df_development['Y1']!=df_development['area'].unique()[1]) & (df_development['Domain']!=df_development['area'].unique()[1]) & (df_development['Abstract']!='-')]
df_evaluation_no_missing = df_evaluation.loc[(df_evaluation['Y2']!=-1) & (df_evaluation['Y']!=-1) & (df_evaluation['Y1']!=df_evaluation['area'].unique()[1]) & (df_evaluation['Domain']!=df_evaluation['area'].unique()[1])& (df_evaluation['Abstract']!='-')]
df_train_no_missing['keywords'].unique()


# In[20]:


df_train_no_missing['Abstract'].unique()


# As just "Keywords" and "Abstract" is going to be use as input and "Domain" as output it's easier to just create a new DataFrame with the usuful data. 

# In[21]:


#train usefull data
df_train_us= df_train_no_missing.drop(['Y2','Y','Y1','area'],axis=1).copy()
df_dev_us = df_development_no_missing.drop(['Y2','Y','Y1','area'],axis=1).copy()
df_eva_us = df_evaluation_no_missing.drop(['Y2','Y','Y1','area'],axis=1).copy()
#df_train_us=X = df_no_missing_train.drop('Y',axis=1).copy()
#df_train_us=X = df_no_missing_train.drop('Y1',axis=1).copy()
#df_train_us=X = df_no_missing_train.drop('area',axis=1).copy()
df_train_us.head()


# The data is now separated in the one it's going to be given as input and as output

# In[22]:


# Domain instead of DOMAIN (LUISJA)
y_train = df_train_us['Domain'].copy()
X_train = df_train_us.drop('Domain',axis=1).copy()
y_dev = df_dev_us['Domain'].copy()
X_dev = df_dev_us.drop('Domain',axis=1).copy()
y_eva = df_eva_us['Domain'].copy()
X_eva = df_eva_us.drop('Domain',axis=1).copy()


# ## Preprocesing data

# Since here the next step is to modified the data in order to be analyzed

# In[23]:


len(X_dev)


# The following section was obtained using a python tutorial:
# https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe

# In[24]:


stopWords_df = pd.read_csv('stopwords_English.txt')   
stopWords = stopWords_df.values.tolist()             
stopWords = list(map(str, chain.from_iterable(stopWords))) 
pat = r'\b(?:{})\b'.format('|'.join(stopWords))       
X_train['clean_abs'] = X_train['Abstract'].str.replace(pat, '')
X_dev['clean_abs'] = X_dev['Abstract'].str.replace(pat, '')
X_eva['clean_abs'] = X_eva['Abstract'].str.replace(pat, '')
#clean .
X_train['clean_abs'] = X_train['clean_abs'].str.replace('.', '')
X_dev['clean_abs'] = X_dev['clean_abs'].str.replace('.', '')
X_eva['clean_abs'] = X_eva['clean_abs'].str.replace('.', '')
#clean ,
X_train['clean_abs'] = X_train['clean_abs'].str.replace(',', '')
X_dev['clean_abs'] = X_dev['clean_abs'].str.replace(',', '')
X_eva['clean_abs'] = X_eva['clean_abs'].str.replace(',', '')
#clean '
X_train['clean_abs'] = X_train['clean_abs'].str.replace("'", '')
X_train['clean_abs'] = X_train['clean_abs'].str.replace('hdph-rs', '')
X_dev['clean_abs'] = X_dev['clean_abs'].str.replace("'", '')
X_eva['clean_abs'] = X_eva['clean_abs'].str.replace("'", '')


# Now clean_abstract has all the abstract but without the stopwords in the .txt.

# In[25]:


X_train_new=X_train.drop('Abstract',axis=1).copy()
X_dev_new=X_dev.drop('Abstract',axis=1).copy()
X_eva_new=X_eva.drop('Abstract',axis=1).copy()
X_eva_new.head()


# In the cell below the vocabulary that is usefull for the analysis will be used and the rest will be erased. This part was also done using a python tutorial.

# In[26]:


# THIS TAKES A LOT OF TIME (LUISJA)

vocab_df = pd.read_csv('vocab.txt')   ## load vocab
vocab = vocab_df.values.tolist()              ## convert df to list
vocab = list(map(str, chain.from_iterable(vocab))) ## join list


#since here it should be different
X_train_new['NAbs']=X_train_new['clean_abs']
X_train_new['NAbs']=X_train_new['NAbs'].apply(lambda x: list(set(x.split())))
X_train_new['vocab']=X_train_new['NAbs'].apply(lambda x: list(set(vocab) & set(x)))

print('done train')

X_dev_new['NAbs']=X_dev_new['clean_abs']
X_dev_new['NAbs']=X_dev_new['NAbs'].apply(lambda x: str(x))
X_dev_new['NAbs']=X_dev_new['NAbs'].apply(lambda x: list(set(x.split())))
X_dev_new['vocab']=X_dev_new['NAbs'].apply(lambda x: list(set(vocab) & set(x))) 

print('done dev')

X_eva_new['NAbs']=X_eva_new['clean_abs']
X_eva_new['NAbs']=X_eva_new['NAbs'].apply(lambda x: str(x))

X_eva_new['NAbs']=X_eva_new['NAbs'].apply(lambda x: list(set(x.split())))
X_eva_new['vocab']=X_eva_new['NAbs'].apply(lambda x: list(set(vocab) & set(x)))
##create a regex


# Finally all the data was cleaned and just the vocabulary needed stayed in the Abstract files

# In[27]:


len(y_dev)


# The following cell imports the word to vector dictionary for doing natural lenguage processing and transforms it into a dictionary.
# Also it defines a function that is going to be used later. This function just implements the dictionary in a different format.

# In[28]:


embeddings_index = {}
f = open('glove.6B.50d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#input = string
#output = python array len=50
def dic(s):
    if s in embeddings_index.keys():
        return embeddings_index[s]
    else:
        return np.zeros(50)


# The cell below defines 1 column for the array of vectors that every word has

# In[29]:


#train part
X_train_new['vec']=X_train_new['vocab'].apply(lambda x: list(map(str.lower, x)))
X_train_new['vec']=X_train_new['vec'].apply(lambda x: list(map(dic,x)) )
#dev part
X_dev_new['vec']=X_dev_new['vocab'].apply(lambda x: list(map(str.lower, x)))
X_dev_new['vec']=X_dev_new['vec'].apply(lambda x: list(map(dic,x)) )
#eva part
X_eva_new['vec']=X_eva_new['vocab'].apply(lambda x: list(map(str.lower, x)))
X_eva_new['vec']=X_eva_new['vec'].apply(lambda x: list(map(dic,x)) )


# All the vectors are analyze

# Finally

# In[30]:


#train part
X_train_new['finAbs']=X_train_new['vec'].apply(lambda x: np.array(x)) 
X_train_new['finAbs']=X_train_new['finAbs'].apply(lambda x: x.sum(axis=0)/len(x)) 
#dev part
X_dev_new['finAbs']=X_dev_new['vec'].apply(lambda x: np.array(x)) 
X_dev_new['finAbs']=X_dev_new['finAbs'].apply(lambda x: x.sum(axis=0)/len(x))
#eva part
X_eva_new['finAbs']=X_eva_new['vec'].apply(lambda x: np.array(x)) 
X_eva_new['finAbs']=X_eva_new['finAbs'].apply(lambda x: x.sum(axis=0)/len(x))


# Now that there is column of arrays of 50 floating pint numbers, it's possible to insert them in the SPM but before it's important to do the same with keywords.

# In[31]:


X_train_bk=X_train_new.drop(['clean_abs','NAbs','vocab','vec'],axis=1).copy()
X_dev_bk=X_dev_new.drop(['clean_abs','NAbs','vocab','vec'],axis=1).copy()
X_eva_bk=X_eva_new.drop(['clean_abs','NAbs','vocab','vec'],axis=1).copy()
X_train_bk.head()


# In this section all the ";" are replaced with blank spaces in order to just get clean words.

# In[32]:


X_train_bk['clean_keywords'] = X_train['keywords'].str.replace(';', '')
X_dev_bk['clean_keywords'] = X_dev['keywords'].str.replace(';', '')
X_eva_bk['clean_keywords'] = X_eva['keywords'].str.replace(';', '')


# In[33]:


X_eva_bk['clean_keywords']


# From here is just to do the same procedure than for abstract and get the vectors finally

# In[34]:


X_train_bk['clean_keywords']=X_train_bk['clean_keywords'].apply(lambda x: list(set(x.split())))
X_dev_bk['clean_keywords']=X_dev_bk['clean_keywords'].apply(lambda x: list(set(x.split())))
X_eva_bk['clean_keywords']=X_eva_bk['clean_keywords'].apply(lambda x: list(set(x.split())))


# In[35]:


X_train_bk['vec']=X_train_bk['clean_keywords'].apply(lambda x: list(map(str.lower, x)))
X_train_bk['vec']=X_train_bk['vec'].apply(lambda x: list(map(dic,x)) )

X_dev_bk['vec']=X_dev_bk['clean_keywords'].apply(lambda x: list(map(str.lower, x)))
X_dev_bk['vec']=X_dev_bk['vec'].apply(lambda x: list(map(dic,x)) )

X_eva_bk['vec']=X_eva_bk['clean_keywords'].apply(lambda x: list(map(str.lower, x)))
X_eva_bk['vec']=X_eva_bk['vec'].apply(lambda x: list(map(dic,x)) )


# In[36]:


#train part
X_train_bk['finKw']=X_train_bk['vec'].apply(lambda x: np.array(x)) 
X_train_bk['finKw']=X_train_bk['finKw'].apply(lambda x: x.sum(axis=0)/len(x)) 
X_train_bk['finKw']=X_train_bk['finKw'].apply(list)
#dev part
X_dev_bk['finKw']=X_dev_bk['vec'].apply(lambda x: np.array(x)) 
X_dev_bk['finKw']=X_dev_bk['finKw'].apply(lambda x: x.sum(axis=0)/len(x))
X_dev_bk['finKw']=X_dev_bk['finKw'].apply(list)
#eva part
X_eva_bk['finKw']=X_eva_bk['vec'].apply(lambda x: np.array(x)) 
X_eva_bk['finKw']=X_eva_bk['finKw'].apply(lambda x: x.sum(axis=0)/len(x))
X_eva_bk['finKw']=X_eva_bk['finKw'].apply(list)


# Finally the data is clean and in the format needed por the Analyzis

# In[37]:


l=np.array(X_train_bk['finKw'].values)
type(l[0])


# In[38]:


X_train_finish=X_train_bk.drop(['clean_keywords','vec','keywords',],axis=1).copy()
X_dev_finish=X_dev_bk.drop(['clean_keywords','vec','keywords',],axis=1).copy()
X_eva_finish=X_eva_bk.drop(['clean_keywords','vec','keywords',],axis=1).copy()

X_train_finish.head()


# ### Change into a lecture format
# All the data has been cleaned but the function just accept data that it's in the format of a matrix so the pandas data frame will change into a numpy array of lists.
# It's important to do the same for the 3 Dataframes

# In[39]:


a_train_abs=np.ones(50)
a_train_k=np.ones(50)
for i in range(X_train_finish.shape[0]):
    a_train_abs= np.vstack([a_train_abs,X_train_finish['finAbs'].values[i]])
    a_train_k= np.vstack([a_train_k,X_train_finish['finKw'].values[i]])
a_train_abs = np.delete(a_train_abs,0,0)
a_train_k = np.delete(a_train_k,0,0)


# In[40]:


a_dev_abs=np.ones(50)
a_dev_k=np.ones(50)
for i in range(X_dev_finish.shape[0]):
    a_dev_abs= np.vstack([a_dev_abs,X_dev_finish['finAbs'].values[i]])
    a_dev_k= np.vstack([a_dev_k,X_dev_finish['finKw'].values[i]])
a_dev_abs = np.delete(a_dev_abs,0,0)
a_dev_k = np.delete(a_dev_k,0,0)


# In[41]:


a_eva_abs=np.ones(50)
a_eva_k=np.ones(50)
for i in range(X_eva_finish.shape[0]):
    a_eva_abs= np.vstack([a_eva_abs,X_eva_finish['finAbs'].values[i]])
    a_eva_k= np.vstack([a_eva_k,X_eva_finish['finKw'].values[i]])
a_eva_abs = np.delete(a_eva_abs,0,0)
a_eva_k = np.delete(a_eva_k,0,0)


# In[42]:


len(a_dev_k)


# ## Supervised vector machine
# In this section the data already procesed is inserted into the supervised vector machine function and it's performance is checked
# The 1st set is just using the abstract so we will compare what happens using abstract, keywords and finally abstract + keywords

# In[43]:


clf_svm2 = SVC(random_state=42)
clf_svm2.fit(a_train_abs, y_train)


# In[44]:


len(a_dev_abs)


# ### Abstract
# The confussion matrix will show the performance of the set.

# In[45]:


plot_confusion_matrix(clf_svm2,a_dev_abs,y_dev,values_format='d')


# ### Keywords
# Same plot for keywords

# In[46]:


clf_svm2 = SVC(random_state=42)
clf_svm2.fit(a_train_k, y_train)


# In[47]:


plot_confusion_matrix(clf_svm2,a_dev_k,y_dev,values_format='d')


# ### Abstract + Keywords
# Finally the abstract will be sumed with the keywords so the checked its performances combined

# In[48]:


sum_train=(a_train_abs + a_train_k)/2
sum_dev=(a_dev_abs + a_dev_k)/2
sum_eva=(a_eva_abs + a_eva_k)/2


# In[59]:


# LUISJA
concat_train = np.concatenate((a_train_abs, a_train_k), axis=1)
concat_dev = np.concatenate((a_dev_abs, a_dev_k), axis=1)
concat_eva = np.concatenate((a_eva_abs, a_eva_k), axis=1)
concat_train.shape


# In[60]:


# LUISJA
clf_svm_CONCAT = SVC(random_state=42)
clf_svm_CONCAT.fit(concat_train, y_train)


# In[61]:


# LUISJA
plot_confusion_matrix(clf_svm_CONCAT,concat_dev,y_dev,values_format='d')


# In[64]:


# LUISJA
ACC = (484+270+364+172+1148+466+363)/len(concat_dev)
ACC


# In[62]:


clf_svm3 = SVC(random_state=42)
clf_svm3.fit(sum_train, y_train)


# In[63]:


plot_confusion_matrix(clf_svm3,sum_dev,y_dev,values_format='d')


# In[53]:


ACC= ( 471+263+357+163+1147+453+368)/len(a_dev_abs) # FIXED by LUISJA
ACC


# The accuracy is 72%

# ## Optimization
# Since there is already a working SVM the goal now is to get better parameters. Even tough the result is quite good changing the parameters can get a better one.

# In[65]:


param_grid= [
    {'C': [0.5, 1, 10, 100],
     'gamma':['scale',1,0.1,0.01,0.001,0.0001],
     'kernel' : ['rbf']},
    ]


# In[66]:


optimal_params= GridSearchCV(SVC(), param_grid, cv=5,scoring='accuracy',verbose=0)


# In[67]:


optimal_params.fit(sum_train, y_train)


# In[68]:


print(optimal_params.best_params_)


# The set parameters in the function SVC of sklearn are C=1, kernel=rbf and gamma=scale. So actually the only one that could be different is gamma

# In[69]:


clf_svm = SVC(random_state=42,C=1,gamma=1)


# In[70]:


clf_svm.fit(sum_train, y_train)


# In[57]:


plot_confusion_matrix(clf_svm,sum_dev,y_dev,values_format='d')


# In[58]:


(475+279+365+193+1159+475+380)/len(a_dev_abs)


# ## Conclusion 
# In this work, successfully a support vector machine was developed in order to analyze the papers given. Clearly the best result is given with the combination of abstract and keyword. Finally it's possible to appreciate that the optimization changing gamma gives an slightly better result. 

# In[ ]:




