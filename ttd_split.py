import random

# Set the percentage split for train, test, and dev sets
train_pct = 0.8
test_pct = 0.1
dev_pct = 0.1

# Set the input and output file paths
input_path = "miwiki_data.txt"
train_path = "train.txt"
test_path = "test.txt"
dev_path = "dev.txt"

# Read in the input file
with open(input_path, "r") as input_file:
    sentences = input_file.readlines()

# Shuffle the sentences
random.shuffle(sentences)

# Calculate the number of sentences for each set
num_sentences = len(sentences)
num_train = int(num_sentences * train_pct)
num_test = int(num_sentences * test_pct)
num_dev = num_sentences - num_train - num_test

# Split the sentences into train, test, and dev sets
train_sentences = sentences[:num_train]
test_sentences = sentences[num_train:num_train+num_test]
dev_sentences = sentences[num_train+num_test:]

# Write the sentences to the output files
with open(train_path, "w") as train_file:
    train_file.writelines(train_sentences)

with open(test_path, "w") as test_file:
    test_file.writelines(test_sentences)

with open(dev_path, "w") as dev_file:
    dev_file.writelines(dev_sentences)