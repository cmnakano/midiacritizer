import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Read in the dataset
with open('miwiki_data_cleaned_unique.txt', 'r') as f:
    sentences = f.readlines()

# Define the vocabulary
vowels = ['a', 'e', 'i', 'o', 'u', 'ā', 'ē', 'ī', 'ō', 'ū']
vocabulary = {'<pad>': 0}

# Define a dictionary mapping binary values to diacritic marks
diacritic_dict = {0: '', 1: '\u0304'}

def custom_loss(y_true, y_pred):
    mask = tf.reduce_any(y_true != 0, axis=-1)
    mask = tf.cast(mask, tf.float32)
    loss = tf.keras.losses.binary_crossentropy(y_true[:, :, 1:], y_pred[:, :, 1:])
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return loss

# Loop through each sentence and add unique characters to the vocabulary dictionary
for sentence in sentences:
    sentence = sentence.strip()
    for char in sentence:
        if char not in vocabulary:
            vocabulary[char] = len(vocabulary)

# Define the reverse vocabulary for decoding purposes
reverse_vocabulary = {idx: char for char, idx in vocabulary.items()}

# Loop through each sentence and tokenize it
input_sequences = []
output_sequences = []
for sentence in sentences:
    sentence = sentence.strip()
    input_sequence = []
    output_sequence = []
    for char in sentence:
        if char.lower() in vowels:
            vowel = char.lower()
            if vowel in ['ā', 'ē', 'ī', 'ō', 'ū']:
                vowel = vowel.lower().replace('ā', 'a').replace('ē', 'e').replace('ī', 'i').replace('ō', 'o').replace('ū', 'u')
                input_sequence += [vocabulary[vowel]]
                output_sequence += [[0, 0, 1]]
            else:
                input_sequence += [vocabulary[vowel]]
                output_sequence += [[0, 1, 0]]
        else:
            input_sequence += [vocabulary[char.lower()]]
            output_sequence += [[1, 0, 0]]
    input_sequences.append(input_sequence)
    output_sequences.append(output_sequence)

# Pad the input and output sequences
max_seq_length = max([len(seq) for seq in input_sequences + output_sequences])
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post', value=vocabulary['<pad>'])
padded_output_sequences = pad_sequences(output_sequences, maxlen=max_seq_length, padding='post', value=[0, 0, 0])
for i in range(len(input_sequences)):
    padded_input_sequences[i, :len(input_sequences[i])] = input_sequences[i]
    padded_output_sequences[i, :len(output_sequences[i])] = output_sequences[i]

# Split the data into training and validation sets
train_size = int(len(padded_input_sequences) * 0.8)
train_inputs = padded_input_sequences[:train_size, :-1]
train_outputs = padded_output_sequences[:train_size, 1:]
val_inputs = padded_input_sequences[train_size:, :-1]
val_outputs = padded_output_sequences[train_size:, 1:]

print('train_inputs shape:', train_inputs.shape)
print('train_outputs shape:', train_outputs.shape)
print('val_inputs shape:', val_inputs.shape)
print('val_outputs shape:', val_outputs.shape)

# Define the model
if os.path.exists('diacritization_model.h5'):
    with tf.keras.utils.custom_object_scope({'custom_loss': custom_loss}):
        model = tf.keras.models.load_model('diacritization_model.h5')
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(vocabulary), output_dim=64, input_shape=(None,), mask_zero=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision()])

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model with early stopping
    model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), batch_size=32, epochs=50, callbacks=[early_stopping])

    # Save the model
    model.save('diacritization_model.h5')

# Define a function to diacritize a sentence
def diacritize_sentence(sentence, model, vocabulary):
    reverse_vocabulary = {idx: char for char, idx in vocabulary.items()}

    sentence = sentence.lower()

    # Encode the input sentence
    input_encoded = []
    for char in sentence:
        input_encoded += [vocabulary[char]]

    # Pad the encoded input sentence
    input_padded = pad_sequences([input_encoded], maxlen=max_seq_length, padding='post', value=vocabulary['<pad>'])

    # Diacritize the sentence by predicting the output sequence using the model
    output_padded = model.predict(input_padded)[0]

    # Convert binary values into integer labels using a threshold
    output_encoded = []
    for token in output_padded:
        if token[2] > max(token[0], token[1]):
            output_encoded.append([0, 0, 1])  # Vowel with macron
        elif token[0] > max(token[1], token[2]):
            output_encoded.append([1, 0, 0])  # Non-vowel character
        elif token[1] > max(token[0], token[2]):
            output_encoded.append([0, 1, 0])  # Vowel without macron
        else:
            # If none of the probabilities are above the threshold, assign the label with the highest probability
            label = [0, 0, 0]
            max_index = np.argmax(token)
            label[max_index] = 1
            output_encoded.append(label)

    # Convert integer labels into diacritic marks
    diacritic_marks = []
    for i, label in enumerate(output_encoded):
        if label == [0, 0, 1]:
            diacritic_marks.append('\u0304')  # Vowel with macron
        elif label == [0, 1, 0]:
            diacritic_marks.append('')  # Vowel without macron
        else:
            diacritic_marks.append('')  # Non-vowel character

    # Combine diacritic marks with input sequence to create output sentence
    output_sequence = []
    print(diacritic_marks[12])
    for i, val in enumerate(input_encoded):
        print(i)
        print(val)
        if i == len(input_encoded)-1:
            break # End of sentence
        elif val in [vocabulary[char] for char in 'aeiou']:
            output_sequence.append(reverse_vocabulary[val] + diacritic_marks[i])
        else:
            output_sequence.append(reverse_vocabulary[val])

    output_sentence = ''.join(output_sequence)

    return output_sentence

# Define an example sentence
sentence = "Ko te whare tenei o nga tangata katoa"

# Diacritize the sentence
diacritized_sentence = diacritize_sentence(sentence, model, vocabulary)

# Print the original and diacritized sentences
print(f"Original sentence: {sentence}")
print(f"Diacritized sentence: {diacritized_sentence}")

# Loop indefinitely to diacritize user input
while True:
    sentence = input('Enter a Maori sentence to diacritize: ')
    diacritized_sentence = diacritize_sentence(sentence, model, vocabulary)
    print('Diacritized sentence:', diacritized_sentence)