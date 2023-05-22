import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

max_seq_length = 680

def replace_macrons(input_str):
    macron_mapping = {
        'ā': 'a',
        'ē': 'e',
        'ī': 'i',
        'ō': 'o',
        'ū': 'u'
    }
    pattern = re.compile('|'.join(macron_mapping.keys()))
    output_str = pattern.sub(lambda x: macron_mapping[x.group()], input_str)
    return output_str

# Define the special tokens
start_token = '<start>'
end_token = '<end>'
pad_token = '<pad>'

# Load the data
with open('miwiki_cleaned.txt', 'r', encoding='utf-8') as f:
    sentences = f.read().splitlines()

# Define the regular expression for splitting sentences into words
word_regex = re.compile(r'\b\w+\b')

# Build the vocabulary
vocabulary = {start_token: 1, end_token: 2, pad_token: 0}
for sentence in sentences:
    for char in sentence:
        vocabulary[char] = vocabulary.get(char, len(vocabulary))

# Encode the sentences into integer sequences
input_encoded_sentences = []
target_encoded_sentences = []
for sentence in sentences:
    input_sentence = replace_macrons(sentence)
    input_encoded_sentence = []
    target_encoded_sentence = []
    for word in word_regex.findall(input_sentence):
        for char in word:
            if char.lower() in ['a', 'e', 'i', 'o', 'u']:
                input_encoded_sentence.append(vocabulary[start_token])
                input_encoded_sentence.append(vocabulary[char])
                input_encoded_sentence.append(vocabulary[end_token])
            else:
                for _ in range(len(char)):
                    input_encoded_sentence.append(vocabulary[char])

    for word in word_regex.findall(sentence):
        for char in word:
            if char.lower() in ['a', 'e', 'i', 'o', 'u', 'ā', 'ē', 'ī', 'ō', 'ū']:
                target_encoded_sentence.append(vocabulary[start_token])
                target_encoded_sentence.append(vocabulary[char])
                target_encoded_sentence.append(vocabulary[end_token])
            else:
                target_encoded_sentence.append(vocabulary[char])

    input_encoded_sentences.append(input_encoded_sentence)
    target_encoded_sentences.append(target_encoded_sentence)

# Pad the encoded sentences
max_seq_length = max(len(seq) for seq in input_encoded_sentences + target_encoded_sentences)
input_padded_sequences = pad_sequences(input_encoded_sentences, maxlen=max_seq_length, padding='post', value=vocabulary[pad_token])
target_padded_sequences = pad_sequences(target_encoded_sentences, maxlen=max_seq_length, padding='post', value=vocabulary[pad_token])

# Save the vocabulary and encoded data
np.save('vocabulary.npy', vocabulary)
np.save('encoded_input_data.npy', input_padded_sequences)

num_examples = len(input_padded_sequences)

# Shuffle the indices of the dataset
indices = np.arange(num_examples)
np.random.shuffle(indices)

# Split the indices into train, validation, and test sets
train_indices = indices[:int(0.7 * num_examples)]
val_indices = indices[int(0.7 * num_examples):int(0.85 * num_examples)]
test_indices = indices[int(0.85 * num_examples):]

# Assign the input and target sequences to each set
train_encoder_input_data = input_padded_sequences[train_indices]
train_decoder_input_data = target_padded_sequences[train_indices][:, :-1]
train_decoder_target_data = tf.one_hot(target_padded_sequences[train_indices][:, 1:], len(vocabulary))

val_encoder_input_data = input_padded_sequences[val_indices]
val_decoder_input_data = target_padded_sequences[val_indices][:, :-1]
val_decoder_target_data = tf.one_hot(target_padded_sequences[val_indices][:, 1:], len(vocabulary))

test_encoder_input_data = input_padded_sequences[test_indices]
test_decoder_input_data = target_padded_sequences[test_indices][:, :-1]
test_decoder_target_data = tf.one_hot(target_padded_sequences[test_indices][:, 1:], len(vocabulary))

# Define the maximum input and output sequence lengths
max_input_seq_length = train_encoder_input_data.shape[1]
max_output_seq_length = train_decoder_input_data.shape[1]

# Define the encoder input
encoder_inputs = Input(shape=(None,))

# Define the encoder embedding layer
encoder_embedding = Embedding(input_dim=len(vocabulary), output_dim=256)(encoder_inputs)

# Define the encoder LSTM layer
encoder_lstm = LSTM(units=256, return_state=True)

# Get the encoder outputs and states
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Keep only the states
encoder_states = [state_h, state_c]

# Define the decoder input
decoder_inputs = Input(shape=(None,))

# Define the decoder embedding layer
decoder_embedding = Embedding(input_dim=len(vocabulary), output_dim=256)

# Get the decoder embeddings
decoder_embeddings = decoder_embedding(decoder_inputs)

# Define the decoder LSTM layer
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# Get the decoder outputs and states
decoder_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)

# Define the output layer
output_layer = Dense(units=len(vocabulary), activation='softmax')

# Get the output probabilities
output_probs = output_layer(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], output_probs)

# Define the loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model with early stopping
model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data, 
          batch_size=64, epochs=10, 
          validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
          callbacks=[early_stopping],
          verbose=2)

# Define the inference encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Define the inference decoder model
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embeddings, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = output_layer(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Save the trained model
model.save('diacritizer.h5')

# Define a function to diacritize a single sentence
def diacritize_sentence(sentence):
    # Encode the sentence
    encoded_sentence = []
    non_vowel_chars = []
    for char in sentence:
        if char.lower() in ['a', 'e', 'i', 'o', 'u']:
            encoded_sentence.append(vocabulary[start_token])
            encoded_sentence.append(vocabulary[char])
            encoded_sentence.append(vocabulary[end_token])
        else:
            encoded_sentence.append(vocabulary[char])
            non_vowel_chars.append(char)

    # Pad the encoded sentence
    padded_sentence = pad_sequences([encoded_sentence], maxlen=max_input_seq_length, padding='post', value=pad_token)

    # Get the encoder states
    states = encoder_model.predict(padded_sentence)

    # Initialize the target sequence with zeros
    target_seq = np.zeros((1, max_output_seq_length))

    # Set the appropriate positions to start and end tokens
    i = 0
    for char in sentence:
        if char.lower() in ['a', 'e', 'i', 'o', 'u']:
            target_seq[0, i] = vocabulary[start_token]
            target_seq[0, i+2] = vocabulary[end_token]
            i += 3
        else:
            i += 1

    # Diacritize the sequence
    diacritized_seq = []
    for i in range(max_output_seq_length):
        # Get the output probabilities and decoder states
        output_probs, h, c = decoder_model.predict([target_seq] + states)

        # Get the most likely output token
        output_token = np.argmax(output_probs[0, i, :])

        # Check if the current token is a start or end token and skip it if it is
        if index_to_token[output_token] in [start_token, end_token]:
            continue

        # Stop if the end token is reached
        if index_to_token[output_token] == pad_token:
            break

        # Append the output token to the diacritized sequence
        diacritized_seq.append(output_token)

        # Update the target sequence and decoder states
        target_seq[0, i] = output_token
        states = [h, c]

    # Decode the diacritized sequence and non-vowel characters
    diacritized_sentence = ''
    j = 0
    for token in diacritized_seq:
        diacritized_char = index_to_token[token]
        while j < len(non_vowel_chars) and diacritized_char == ' ':
            diacritized_sentence += non_vowel_chars[j]
            j += 1
        diacritized_sentence += diacritized_char

    # Append any remaining non-vowel characters to the end of the diacritized sentence
    while j < len(non_vowel_chars):
        diacritized_sentence += non_vowel_chars[j]
        j += 1

    # Return the diacritized sentence
    return diacritized_sentence

# Test the model on some example sentences
test_sentences = [
"Ko te whare tenei o nga tangata katoa",
"He kai kei aku ringa",
"Ka pu te ruha, ka hao te rangatahi",
"He whare tino whakahirahira tenei mo toku iwi",
"Kua tae mai te wa",
"Ma te wa ka whakatomuri ai",
"E kore e taea e ia te hu",
"Whakapau kaha ki te whai i ou moemoea",
"He waka eke noa",
"Whaia te iti kahurangi ki te tuohu koe me he maunga teitei"
]

for sentence in test_sentences:
    diacritized_sentence = diacritize_sentence(sentence)
    print(f"Original sentence: {sentence}")
    print(f"Diacritized sentence: {diacritized_sentence}\n")