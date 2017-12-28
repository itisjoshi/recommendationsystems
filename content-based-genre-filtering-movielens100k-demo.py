
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


cols = ['mid', 'title' ,'mov-release-date','vid-release-date', 
        'imdb', 'unknown', 'action', 'adventure', 'animation', 
        'childrens', 'comedy', 'crime', 'documentary', 'drama', 
        'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 
        'romance', 'sci-fi', 'thriller', 'war', 'western']
item_df_details = pd.read_csv('ml-100k/u.item',
                              sep='|', names=cols, index_col='mid', 
                              encoding='latin-1')
item_df = item_df_details.drop(
    ['mov-release-date','vid-release-date','imdb','title'], axis=1)


# In[3]:


from sklearn.metrics.pairwise import pairwise_distances
item_dist = pairwise_distances(item_df, metric='hamming')
item_sim = np.subtract(np.ones(item_dist.shape), item_dist)


# In[4]:


item_sim


# In[5]:


# toy-story and aladdin-and-the-king-of-thieves have the same genres, should have 1.0 similarity
item_sim[0,421]


# In[6]:


cols = ['uid', 'mid', 'rating', 'timestamp']
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=cols)


# In[7]:


ratings_df.loc[1]


# In[8]:


# hierarchical index, so we can see all the movie ratings for a user 
ratings_df = ratings_df.set_index('uid','mid')


# In[9]:


def recommend(df, item_sim, uid):
    user_watched = ratings_df.loc[uid].mid.tolist()
    highest_sim = -1.0
    most_sim = -1
    for uid, mid in df.loc[uid][df.loc[uid].rating==5]['mid'].iteritems():
        sim_items = item_sim[mid-1]
        sim_items[mid-1] = 0.0
        max_ix = np.argmax(sim_items)
        if max_ix+1 in user_watched:
            continue
        if item_sim[mid-1,max_ix] > highest_sim:
            highest_sim = item_sim[mid-1,max_ix]
            most_sim = max_ix+1
    return most_sim


# In[11]:


uid = 200
item_df_details.loc[recommend(ratings_df, item_sim, uid)].title

