
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


cols = ['uid', 'mid', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=cols)


# In[3]:


n_users = df.uid.unique().shape[0]
n_items = df.mid.unique().shape[0]


# In[4]:


from sklearn import cross_validation
train_data_df, test_data_df = cross_validation.train_test_split(df, test_size=0.20)


# In[5]:


train_data_df.sample(5)


# In[6]:


# l: Pandas(Index=9218, uid=271, mid=15, rating=3, timestamp=885847876)
train_data = np.zeros((n_users, n_items))
for l in train_data_df.itertuples():
    train_data[l.uid-1, l.mid-1] = l.rating

test_data = np.zeros((n_users, n_items))
for l in test_data_df.itertuples():
    test_data[l.uid-1, l.mid-1] = l.rating


# In[7]:


from sklearn.metrics.pairwise import pairwise_distances
user_sim = pairwise_distances(train_data, metric='cosine')
item_sim = pairwise_distances(train_data.T, metric='cosine')


# In[8]:


def predict_user_based(ratings, sim):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    return mean_user_rating[:, np.newaxis] + sim.dot(ratings_diff)         / np.array([np.abs(sim).sum(axis=1)]).T


# In[9]:


def predict_item_based(ratings, sim):
    return ratings.dot(sim) / np.array([np.abs(sim).sum(axis=1)])


# In[10]:


item_pred = predict_item_based(train_data, item_sim)
user_pred = predict_user_based(train_data, user_sim)


# In[11]:


from sklearn.metrics import mean_squared_error as mse
from math import sqrt
def rmse(pred, truth):
    pred = pred[truth.nonzero()].flatten()
    truth = truth[truth.nonzero()].flatten()
    return sqrt(mse(pred, truth))


# In[14]:


print('user-user CF rmse: ' + str(rmse(user_pred, test_data)))
print('item-item CF rmse: ' + str(rmse(item_pred, test_data)))


# In[13]:


(1.0-len(df)/float(n_users*n_items)) * 100.0


# In[14]:


import scipy.sparse as sp
from scipy.sparse.linalg import svds


# In[15]:


#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data, k=10)
s_diag = np.diag(s)
x_pred = np.dot(np.dot(u, s_diag), vt)
print 'svd-fixed CF rmse: ' + str(rmse(x_pred, test_data))

