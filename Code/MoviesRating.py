# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:05:06 2022

@author: razze
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import colors
from Matcher import Matcher
import train_parameters
import model_parameters
import ot
import tensorflow as tf
import seaborn as sns


header = ['userId','movieId','rating','timestamp']
dataset = pd.read_csv("dataset2/ratings.csv", names=header)

coulmn = ['movieId',' movieTitle','genre'] 

items_dataset = pd.read_csv('dataset2/movies.csv',names=coulmn)
merged_dataset = pd.merge(dataset, items_dataset, how='inner', on='movieId')

refined_dataset = merged_dataset.groupby(by=['userId','movieId'], as_index=False).agg({"rating":"mean"})
refined_dataset=refined_dataset[refined_dataset['userId']<11]
num_users = len(refined_dataset['userId'].value_counts())
num_items = len(refined_dataset['movieId'].value_counts())
print('Unique number of users in the dataset: {}'.format(num_users))
print('Unique number of movies in the dataset: {}'.format(num_items))

rating_count_df = pd.DataFrame(refined_dataset.groupby(['rating']).size(), columns=['count'])
rating_count_df
Nij= rating_count_df.to_numpy()
N= np.sum(Nij)

rating_count_df1 = pd.DataFrame(refined_dataset.groupby(['userId'])['rating'].sum('rating'))
rating_count_df1.sort_values('userId', ascending=False)
ratingsByUser=rating_count_df1.iloc[:,:1].values

rating_count_df2 = pd.DataFrame(refined_dataset.groupby(['movieId']).size(), columns=['count'])
rating_count_df2.sort_values('count', ascending=False)

ax = rating_count_df.reset_index().rename(columns={'index': 'rating score'}).plot('rating','count', 'bar',
    figsize=(12, 8),
    title='Count for Each Rating Score',
    fontsize=12)

ax.set_xlabel("movie rating score")
ax.set_ylabel("number of ratings")

total_count = num_items * num_users
zero_count = total_count-refined_dataset.shape[0]

rating_count_df = rating_count_df.append(
    pd.DataFrame({'count': zero_count}, index=[0.0]),
    verify_integrity=True,
).sort_index()

user_to_movie_df = refined_dataset.pivot(
    index='userId',
     columns='movieId',
      values='rating').fillna(0)

user_to_movie_df=user_to_movie_df.to_numpy()
m= np.empty((user_to_movie_df.shape[0],user_to_movie_df.shape[1]))
for i in range(user_to_movie_df.shape[0]):
     for j in range(user_to_movie_df.shape[1]):
        if (user_to_movie_df[i][j] != 0 ):
            m[i][j]=rating_count_df.loc[user_to_movie_df[i][j]][0]/N 
        else:
            m[i][j]=0
m1=m

U = np.array(user_to_movie_df).T
V = np.eye(user_to_movie_df.shape[1])

p, m = U.shape
q, n = V.shape
r = user_to_movie_df.shape[0]

seed = user_to_movie_df.shape[1]
rng = np.random.RandomState(seed)
G0 = rng.rand(r, p)
D0 = rng.rand(r, q)
A0 = np.dot(G0.T, D0)

model = Matcher(pi_sample=m1, U0=U, V0=V, r=r)

train_param = train_parameters(max_outer_iteration=2, max_inner_iteration=2, learning_rate=1)

model_param = model_parameters(A0=A0, gamma=0.2, const=1, degree=2, lam=0.001, lambda_mu=1, lambda_nu=1, delta=0.005)

optimal = model.riot(model_param=model_param, train_param=train_param)




