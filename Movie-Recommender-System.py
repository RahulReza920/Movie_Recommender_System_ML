#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 


# In[4]:


movies.head()


# In[5]:


movies.head(2)


# In[6]:


movies.info()


# In[7]:


credits.head()


# In[8]:


movies = movies.merge(credits,on='title')


# In[9]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


import ast


# In[12]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[15]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[16]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[17]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[18]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[19]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[20]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[21]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[22]:


movies.head()


# In[23]:


movies.sample(5)


# In[24]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[25]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[26]:


movies.head()


# In[27]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[28]:


movies.head()


# In[29]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[30]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[31]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[32]:


new['tags'][0]


# In[34]:


new['tags']=new['tags'].apply(lambda x:x.lower())


# In[35]:


new.head()


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[37]:


vector = cv.fit_transform(new['tags']).toarray()


# In[38]:


vector.shape


# In[39]:


from sklearn.metrics.pairwise import cosine_similarity


# In[40]:


similarity = cosine_similarity(vector)


# In[41]:


similarity


# In[42]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[43]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[44]:


recommend('Gandhi')


# In[45]:


import pickle


# In[46]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[47]:


recommend(Batman Begin)


# In[53]:


recommend('Titanic')


# In[ ]:




