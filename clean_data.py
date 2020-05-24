# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Setup
# %% [markdown]
# ## Import and check packages

# %%
import sys
import json
import pandas as pd
import numpy as np
import os
import csv
import re, string


# %%
print('Version check:')
print('Python: {}'.format(sys.version))
print('pandas: {}'.format(pd.__version__))
print('regex: {}'.format(re.__version__))

# %% [markdown]
# ## Functions

# %%
def parse_file(path):
    data = list()
    with open(path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            data.append(row)

    return data

# %% [markdown]
# # Import and prepare data

# %%
path = '/home/specc/Documents/school_files/453_nlp/assignment03_files/data_files/'
os.chdir(path) # set path to data files
len(os.listdir()) # check how many files are present


# %%
df = pd.DataFrame()
for item in os.listdir():
    test_path = item
    data = parse_file(item)
    temp_df = pd.DataFrame(data[1:], columns=data[0])
    df = pd.concat([df, temp_df], ignore_index=0)


# %%
df.head()


# %%
df.tail()

# %% [markdown]
# ## Drop rows with no text review

# %%
start_count = len(df)
start_count


# %%
df[df['Review'].isnull()]


# %%
df.iloc[200]


# %%
df.iloc[500]


# %%
df.columns


# %%
df = df[df['Review'].notnull()] # removing odd data points


# %%
dropped_count = len(df) # comparing to old value


# %%
# difference
rows_removed = start_count - dropped_count
print('Number of blank reviews: {}'.format(rows_removed))

# %% [markdown]
# ## Remove reviews with fewer than 50 words

# %%
## Reviews with less than 50 words
def word_count(row):
    num_words = len(row)

    return num_words


# %%
df['review_word_count'] = df['Review'].apply(word_count)


# %%
df.review_word_count.describe()


# %%
before_len = len(df)


# %%
df = df[df['review_word_count'] >= 50]


# %%
after_len = len(df)


# %%
num_removed = before_len - after_len
print('Number of reviews with fewer than 50 words: {}'.format(num_removed))

# %% [markdown]
# ## Remove reviews with no ratings

# %%
start_count = len(df)


# %%
df = df[df.Rating.notnull()]
df.head()


# %%
end_count = len(df)

num_empty_ratings = start_count - end_count
print('Number of rows with no ratings: {}'.format(num_empty_ratings))

# %% [markdown]
# ## Split train/test set

# %%
shuffle_df = df.sample(frac=1).reset_index(drop=True)


# %%
shuffle_df.head()


# %%
int(len(df) * 0.30)


# %%
int(len(df) * 0.70)


# %%
int(len(df) * 0.30) + int(len(df) * 0.70)


# %%
len(shuffle_df.iloc[:int(len(df) * 0.70)])


# %%
len(shuffle_df.iloc[int(len(df) * 0.70):])


# %%
train_df = shuffle_df.iloc[:int(len(df) * 0.70)]
test_df = shuffle_df.iloc[int(len(df) * 0.70):]

# %% [markdown]
# # Export dataframes

# %%
# train_df.to_pickle('train_df.pkl')
# test_df.to_pickle('test_df.pkl')


# %%
df.to_pickle('processed_data.pkl')

