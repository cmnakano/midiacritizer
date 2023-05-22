import re

# Load the corpus
with open('miwiki_data_cleaned_unique_2.txt', 'r') as file:
    corpus = file.read()

# Preprocess the corpus
corpus = corpus.lower()
corpus = re.sub(r'[^\w\s]', '', corpus)
words = corpus.split()

# Count the unique words and total words
unique_words = set(words)
num_unique_words = len(unique_words)
num_total_words = len(words)

# Print the results
print("Number of unique words in corpus:", num_unique_words)
print("Total number of words in corpus:", num_total_words)