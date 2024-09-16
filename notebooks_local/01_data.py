# %%
"""
# Data

We will first inspect the data to see what we are working with. We will be using the wikitext-raw-2 dataset, which is a collection of Wikipedia articles.
"""

# %%
# Importing the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# %%
os.chdir('../')

# %%
pwd

# %%
from src.utils import *

# %%
"""
Utils contains helper functions for loading, preprocessing, and saving the data.
"""

# %%
PATH_TEST_FILE = 'data_small/raw/test.parquet'

# %%
# Reading the dataset
df = pd.read_parquet(PATH_TEST_FILE)

# %%
text = ''
for line in df['text']:
    text += line

# %%
text[:100]

# %%
# Save as txt file
if not os.path.exists('data_small/raw'):
    os.makedirs('data_small/raw')
with open('data_small/raw/test.txt', 'w') as f:
    f.write(text)

# %%
print(f"Sample text: \n\n {word_wrap(text[:500])}")

# %%
"""
Let's first extract the topics in the dataset.
"""

# %%
main_topics = extract_main_topics(text)

main_topics[:10]

# %%
main_topics

# %%
all_topics = extract_all_topics(text)
all_topics[:10]

# %%
all_topics

# %%
# distribution of topic lengths
sns.set()
sns.displot([len(t) for t in all_topics])
plt.xlabel('Topic length')
plt.ylabel('Density')
plt.title('Distribution of topic lengths')
plt.show()


# %%
"""
We understand that the data is in heierarchial format, so we will inject higher level topics into the lower level topics.

    Example: 
    
    = Topic =
    == Subtopic ==
    === Subsubtopic ===

    will be converted to:
    = Topic =
    = Topic-Subtopic =
    = Topic-Subtopic-Subsubtopic =


"""

# %%
new_text = modify_topics(text)

# %%
new_main_topics = extract_main_topics(new_text)

# %%
# All topics
new_all_topics = extract_all_topics(new_text)

# %%
all_topics[:10]

# %%
new_all_topics[:10]

# %%
sns.displot([len(t) for t in new_all_topics])
plt.xlabel('Topic length')
plt.ylabel('Density')
plt.title('Distribution of topic lengths')
plt.show()

# %%
# topics over 100 characters
long_topics = [t for t in new_all_topics if len(t) > 100]
print(f'There are {len(long_topics)} topics over 500 characters long.')

# %%
long_topics

# %%
len(long_topics)

# %%
search_string = 'Thus , k will have units of impedance , that is , ohms . It is readily apparent that in order for k to be constant ,'
print(word_wrap(extract_chars_around_string(text, search_string, 50, 500)))

# %%
print(word_wrap(extract_chars_around_string(text, search_string, 100, 50)))

# %%
print(word_wrap(extract_chars_around_string(text, search_string, 0, 0)))

# %%
topic_char_counts = count_chars_in_topics(new_text)

# %%
# plot the distribution of characters in topics
sns.displot([count for count in topic_char_counts.values()])
plt.xlabel('Number of characters')
plt.ylabel('Density')
plt.title('Distribution of characters in topics')
plt.show()


# %%
"""
---
"""