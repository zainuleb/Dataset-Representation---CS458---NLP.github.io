#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from sklearn.feature_extraction.text import CountVectorizer

corpus=open('Movies_TV.txt').read()


# In[3]:


#Removing first line.
res=re.sub(r'Domain.*\n','',corpus)
data_list=res.split('\t')


# In[11]:


#2) Structure with Frequency

f_vect=CountVectorizer()
F=f_vect.fit_transform(data_list)
#print(f_vect.vocabulary_)


# In[12]:


#2) Structure with binary

vect=CountVectorizer(binary=True)
X=vect.fit_transform(data_list)
#X.toarray()


# In[13]:


#2) Structure with tfidf based representation

from sklearn.feature_extraction.text import TfidfTransformer
vect2=TfidfTransformer()
Y=vect2.fit_transform(X)
#Y.toarray()

