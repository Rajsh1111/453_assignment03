# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Setup
# %% [markdown]
# ## Import and check packages

# %%
import sys
import pandas as pd
import numpy as np
import nltk
import gensim
import sklearn
import re, string
from nltk.stem import PorterStemmer
import multiprocessing
import os
import csv
import matplotlib
import scipy


# %%
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,                                             CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


# %%
print('Version check:')
print('Python: {}'.format(sys.version))
print('pandas: {}'.format(pd.__version__))
print('nltk: {}'.format(nltk.__version__))
print('gensim: {}'.format(gensim.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('regex: {}'.format(re.__version__))
print('scipy: {}'.format(scipy.__version__))

# %% [markdown]
# ## Import data

# %%
# train_df = pd.read_pickle('data_files/train_df.pkl')
# train_df.head()


# %%
# test_df = pd.read_pickle('data_files/test_df.pkl')
# test_df.head()


# %%
df = pd.read_pickle('data_files/processed_data.pkl')
shuffled_df = df.sample(frac=1)
shuffled_df.head()


# %%
shuffled_df.dtypes


# %%
shuffled_df.Rating = shuffled_df.Rating.astype('float')


# %%
shuffled_df.dtypes


# %%
shuffled_df['rounded_rating'] = shuffled_df.Rating.round(0)


# %%
shuffled_df.head()


# %%
shuffled_df.columns


# %%
shuffled_df.rounded_rating.value_counts()

# %% [markdown]
# ## EDA

# %%
shuffled_df.rounded_rating.hist()


# %%
shuffled_df.Rating.hist()
plt.savefig('ratings_dist.png', format = 'png')


# %%
shuffled_df.review_word_count.hist()
plt.savefig('word_counts_dist.png', format='png')


# %%
shuffled_df.review_word_count.describe()

# %% [markdown]
# ## Setup functions

# %%
def clean_doc(doc): 
    # split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    # #lowercase all words
    tokens = [word.lower() for word in tokens]
    # # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]         
    # # word stemming Commented
    if STEMMING:
        ps=PorterStemmer()
        tokens=[ps.stem(word) for word in tokens]
    return tokens

# %% [markdown]
# ## Analysis settings

# %%
n_dim = 100
n_grams = 1
random_seed = 88
cores = multiprocessing.cpu_count()
STEMMING = True

# %% [markdown]
# ## Sample documents

# %%
shuffled_df = shuffled_df.iloc[:1800]

# %% [markdown]
# # Vectorize Data
# %% [markdown]
# ## Prepare data

# %%
# train_docs = list()
# gensim_train = list()
# for i in range(len(train_df)):
#     temp_text = train_df['Review'].iloc[i]
#     cleaned_doc = clean_doc(temp_text)
#     gensim_train.append(cleaned_doc)
#     #print(temp_text)
#     combined_text = ' '.join(clean_doc(temp_text))
#     train_docs.append(combined_text)


# %%
# test_docs = list()
# gensim_test = list()
# for i in range(len(test_df)):
#     temp_text = train_df['Review'].iloc[i]
#     cleaned_doc = clean_doc(temp_text)
#     gensim_test.append(cleaned_doc)
#     #print(temp_text)
#     combined_text = ' '.join(clean_doc(temp_text))
#     test_docs.append(combined_text)


# %%
docs = list()
gensim_docs = list()

for i in range(len(shuffled_df)):
    temp_text = shuffled_df['Review'].iloc[i]
    cleaned_doc = clean_doc(temp_text)
    gensim_docs.append(cleaned_doc)
    #print(temp_text)
    combined_text = ' '.join(clean_doc(temp_text))
    docs.append(combined_text)

# %% [markdown]
# ## CountVectorizer

# %%
count_vec = CountVectorizer(ngram_range=(1, 1), max_features=100)
count_matrix = count_vec.fit_transform(docs)


# %%
count_matrix[0].toarray()


# %%
print(count_vec.get_feature_names()[:5])

# %% [markdown]
# ## TF-IDF

# %%
tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_features=100)
tfidf_matrix =tfidf_vec.fit_transform(docs)


# %%
tfidf_vec.get_feature_names()[:5]


# %%
tfidf_matrix.T[0].toarray()


# %%
len(tfidf_vec.get_feature_names())

# %% [markdown]
# ## Doc2Vec

# %%
train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(gensim_docs)]
cores = multiprocessing.cpu_count()

print("\nWorking on Doc2Vec vectorization, dimension 100")
model_100 = Doc2Vec(train_corpus, vector_size = 100, window = 4, 
	min_count = 1, workers = cores, epochs = 20)

model_100.train(train_corpus, total_examples = model_100.corpus_count, 
	epochs = model_100.epochs)  # build vectorization model on training set

# vectorization for the training set
doc2vec_100_vectors = np.zeros((len(gensim_docs), 100)) # initialize numpy array
for i in range(0, len(gensim_docs)):
    doc2vec_100_vectors[i,] = model_100.infer_vector(gensim_docs[i]).transpose()
print('\nTraining doc2vec_100_vectors.shape:', doc2vec_100_vectors.shape)
# print('doc2vec_100_vectors[:2]:', doc2vec_100_vectors[:2])

# vectorization for the test set
# doc2vec_100_vectors_test = np.zeros((len(gensim_test), 100)) # initialize numpy array
# for i in range(0, len(gensim_test)):
#     doc2vec_100_vectors_test[i,] = model_100.infer_vector(gensim_test[i]).transpose()
# print('\nTest doc2vec_100_vectors_test.shape:', doc2vec_100_vectors_test.shape)


# %%
type(doc2vec_100_vectors)

# %% [markdown]
# # Clustering Analysis
# %% [markdown]
# ## CountVectorizer

# %%
k=5 # for the number of stars given to each review
km = KMeans(n_clusters=k, random_state =random_seed)
km.fit(count_matrix)
clusters = km.labels_.tolist()

terms = count_vec.get_feature_names()
Dictionary={'Cluster':clusters, 'Text': docs, 'Rating' : shuffled_df.rounded_rating.iloc[:1800]}
frame=pd.DataFrame(Dictionary, columns=['Rating', 'Cluster','Text'])
frame['record']=1
frame.head(n=10)


# %%
frame[frame.Cluster == 3].Rating.mean()


# %%
frame[frame.Cluster == 0].Rating.mean()


# %%
frame[frame.Cluster == 1].Rating.mean()


# %%
frame[frame.Cluster == 2].Rating.mean()


# %%
frame[frame.Cluster == 4].Rating.mean()


# %%
frame.groupby(['Cluster']).mean()


# %%
frame.groupby(['Cluster']).mean().to_excel('countvec_cluster_means.xlsx')


# %%
print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]

#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

# dictionary to store terms and titles
cluster_terms={}
cluster_title={}

for i in range(k):
    print("Cluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms


# %%
pivot=pd.pivot_table(frame, values='record', index='Rating',
                     columns='Cluster', aggfunc=np.sum, fill_value=0)

pivot.head(n=25)


# %%
pivot.to_excel('countvec_pivot.xlsx')

# %% [markdown]
# ## TF-IDF

# %%
###############################################################################
### K Means Clustering - TFIDF
###############################################################################
#### Note: Here we carry out a clustering of the documents
k=5
km = KMeans(n_clusters=k, random_state =random_seed)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

terms = tfidf_vec.get_feature_names()
Dictionary={'Cluster':clusters, 'Text': docs, 'Rating' : shuffled_df.rounded_rating.iloc[:1800]}
frame=pd.DataFrame(Dictionary, columns=['Rating', 'Cluster','Text'])
frame['record']=1
frame.head()


# %%
print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]

#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

# dictionary to store terms and titles
cluster_terms={}
cluster_title={}

for i in range(k):
    print("Cluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms


# %%
pivot=pd.pivot_table(frame, values='record', index='Rating',
                     columns='Cluster', aggfunc=np.sum, fill_value=0)

pivot.to_excel('tfidf_pivot.xlsx')
pivot


# %%
frame[frame.Cluster == 0].Rating.mean()


# %%
frame[frame.Cluster == 1].Rating.mean()


# %%
frame[frame.Cluster == 2].Rating.mean()


# %%
frame[frame.Cluster == 3].Rating.mean()


# %%
frame[frame.Cluster == 4].Rating.mean()


# %%
frame.groupby(['Cluster']).mean()


# %%
frame.groupby(['Cluster']).mean().to_excel('tfidf_cluster_means.xlsx')

# %% [markdown]
# ## Doc2Vec

# %%
k=5
km = KMeans(n_clusters=k, random_state =random_seed)
km.fit(doc2vec_100_vectors)
clusters = km.labels_.tolist()

#terms = tfidf_vec.get_feature_names()
Dictionary={'Cluster':clusters, 'Text': docs, 'Rating' : shuffled_df.rounded_rating.iloc[:1800]}
frame=pd.DataFrame(Dictionary, columns=['Rating', 'Cluster','Text'])
frame['record']=1
frame.head()


# %%
print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

terms_dict=[]

#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

# dictionary to store terms and titles
cluster_terms={}
cluster_title={}

for i in range(k):
    print("Cluster %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms

    # print("Cluster %d titles:" % i, end='')
    # temp=frame[frame['Cluster']==i]
    # for title in temp['Car']:
    #     #print(' %s,' % title, end='')
    #     temp_titles.append(title)
    # cluster_title[i]=temp_titles


# %%
pivot=pd.pivot_table(frame, values='record', index='Rating',
                     columns='Cluster', aggfunc=np.sum, fill_value=0)

pivot.to_excel('docvec_pivot.xlsx')
pivot


# %%
frame[frame.Cluster == 0].Rating.mean()


# %%
frame[frame.Cluster == 1].Rating.mean()


# %%
frame[frame.Cluster == 2].Rating.mean()


# %%
frame[frame.Cluster == 3].Rating.mean()


# %%
frame[frame.Cluster == 4].Rating.mean()


# %%
frame.groupby(['Cluster']).mean()


# %%
frame.groupby(['Cluster']).mean().to_excel('docvec_cluster_means.xlsx')

# %% [markdown]
# # T-SNE Analysis
# %% [markdown]
# ## CountVectorizer

# %%
#### Note: TSNE algorithm used for multidimensional scaling
mds = TSNE(n_components=2, metric="euclidean", random_state=random_seed)

# Note: The objective here is to obtain a picture of the documents in two dimensions
pos = mds.fit_transform(count_matrix.toarray())  # shape (n_components, n_samples)


# %%
xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'grey', 2: 'blue', 3: 'rosybrown', 4: 'firebrick'}

#set up cluster names using a dict.  
cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 
                  4: 'Cluster 4'}

#set up cluster names using a dict.  
#cluster_dict = cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_labels[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #show legend with only 1 point
plt.savefig('countvec_tsne_docs.png', format = 'png')

# %% [markdown]
# ## TF-IDF

# %%
#### Note: TSNE algorithm used for multidimensional scaling
mds = TSNE(n_components=2, metric="euclidean", random_state=random_seed)

# Note: The objective here is to obtain a picture of the documents in two dimensions
pos = mds.fit_transform(tfidf_matrix.toarray())  # shape (n_components, n_samples)


# %%
xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'grey', 2: 'blue', 3: 'rosybrown', 4: 'firebrick'}

#set up cluster names using a dict.  
cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 
                  4: 'Cluster 4'}

#set up cluster names using a dict.  
#cluster_dict = cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_labels[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #show legend with only 1 point
plt.savefig('tfidf_tsne_docs.png', format = 'png')

# %% [markdown]
# ## Doc2Vec

# %%
#### Note: TSNE algorithm used for multidimensional scaling
mds = TSNE(n_components=2, metric="euclidean", random_state=random_seed)

# Note: The objective here is to obtain a picture of the documents in two dimensions
pos = mds.fit_transform(doc2vec_100_vectors)  # shape (n_components, n_samples)


# %%
xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'grey', 2: 'blue', 3: 'rosybrown', 4: 'firebrick'}

#set up cluster names using a dict.  
cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 
                  4: 'Cluster 4'}

#set up cluster names using a dict.  
#cluster_dict = cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 12)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_labels[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #show legend with only 1 point

plt.savefig('docvec_tsne_docs.png', format = 'png')


# %%
for name,group in groups:
    print(name)
    print(group)


# %%
len(clusters)

# %% [markdown]
# # t-SNE on words
# %% [markdown]
# ## CountVectorizer

# %%
#### Note: TSNE algorithm used for multidimensional scaling
mds = TSNE(n_components=2, metric="euclidean", random_state=random_seed)

# Note: The objective here is to obtain a picture of the documents in two dimensions
pos = mds.fit_transform(count_matrix.T.toarray())  # shape (n_components, n_samples)


# %%
# k=5
# km = KMeans(n_clusters=k, random_state =random_seed)
# km.fit(count_matrix.T)
# clusters = km.labels_.tolist()


# %%
xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'grey', 2: 'blue', 3: 'rosybrown', 4: 'firebrick'}

#set up cluster names using a dict.  
cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 
                  4: 'Cluster 4'}

labels = np.array(count_vec.get_feature_names())

#set up cluster names using a dict.  
#cluster_dict = cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=labels)) 

#group by cluster
groups = df.groupby('label')

df.groupby('label')


# %%
fig, ax = plt.subplots(figsize=(15, 15)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            #label=cluster_labels[name], color=cluster_colors[name], 
            mec='none',
            color = 'grey', alpha = 0.5)
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')

    ax.annotate(name, (group.x, group.y),
                textcoords="offset points",
                xytext=(0,10))

plt.savefig('countvec_mds_terms.png', format = 'png')

#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #show legend with only 1 point

# %% [markdown]
# ## TF-IDF

# %%
#### Note: TSNE algorithm used for multidimensional scaling
mds = TSNE(n_components=2, metric="euclidean", random_state=random_seed)

# Note: The objective here is to obtain a picture of the documents in two dimensions
pos = mds.fit_transform(tfidf_matrix.T.toarray())  # shape (n_components, n_samples)


# %%
xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {0: 'black', 1: 'grey', 2: 'blue', 3: 'rosybrown', 4: 'firebrick'}

#set up cluster names using a dict.  
cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 
                  4: 'Cluster 4'}

labels = np.array(tfidf_vec.get_feature_names())

#set up cluster names using a dict.  
#cluster_dict = cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=labels)) 

#group by cluster
groups = df.groupby('label')

df.groupby('label')


# %%
fig, ax = plt.subplots(figsize=(15, 15)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            #label=cluster_labels[name], color=cluster_colors[name], 
            mec='none',
            color='grey',
            alpha = 0.5)
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')

    ax.annotate(name, (group.x, group.y),
                textcoords="offset points",
                xytext=(0,10))
plt.savefig('tfidf_mds_terms.png', format = 'png')

#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #show legend with only 1 point

# %% [markdown]
# # Hierarchical Cluster Analysis
# %% [markdown]
# ## CountVectorizer

# %%
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

dist = pdist(count_matrix.T.toarray())
linked = linkage(dist, method='ward')
fcluster = fcluster(linked, 0, criterion = 'distance')


# %%
linked = linkage(count_matrix.T.toarray(), method='ward')

plt.subplots(figsize=(15, 22)) # set size
dendrogram(linked, 
            labels = count_vec.get_feature_names(),
            orientation='right')

plt.tick_params(axis='y', labelsize=15)
plt.grid(False)
plt.savefig('countvec_dendrogram.png', format = 'png')
plt.show()

# %% [markdown]
# ## TF-IDF

# %%
linked = linkage(tfidf_matrix.T.toarray(), method='ward')

plt.subplots(figsize=(15, 22)) # set size
dendrogram(linked,
            labels = tfidf_vec.get_feature_names(),
            orientation = 'right')
# plt.savefig('C:\\Users\\bxiao\Documents\\school_files\\453_nlp\\assignments\\453_assignment03\\tfidf_dendrogram01.png', 
#                 format = 'png', dpi = 1600)
plt.tick_params(axis='y', labelsize=12)
plt.grid(False)
plt.savefig('tfidf_dendrogram.png', format = 'png')
plt.show()

