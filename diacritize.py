import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the vocabulary
vocabulary = np.load('vocabulary.npy', allow_pickle=True).item()

# Load the encoded input data
input_padded_sequences = np.load('encoded_input_data.npy')

# Get the maximum sequence length
max_seq_length = input_padded_sequences.shape[1]

# Define the special tokens
start_token = '<start>'
end_token = '<end>'
pad_token = '<pad>'

# Load the saved model
model = tf.keras.models.load_model('diacritizer.h5')

# Define a function to diacritize a single sentence
def diacritize_sentence(sentence, model, vocabulary):
    reverse_vocabulary = {idx: char for char, idx in vocabulary.items()}

    sentence = sentence.lower()

    # Encode the input sentence
    input_encoded = []
    for char in sentence:
        if char in 'aeiou':
            input_encoded += [vocabulary[start_token], vocabulary[char], vocabulary[end_token]]
        else:
            input_encoded += [vocabulary[char]]

    # Pad the encoded input sentence
    input_padded = pad_sequences([input_encoded], maxlen=max_seq_length, padding='post', value=vocabulary[pad_token])
    print(input_padded)

    # Diacritize the sentence by predicting the output sequence using the model
    output_padded = model.predict([input_padded, np.zeros((1, max_seq_length))])[0]
    print(output_padded)
    output_encoded = [np.argmax(token) for token in output_padded]
    
    # Find the position of the final end_token in the output sequence
    end_token_positions = [i for i, val in enumerate(output_encoded) if val == vocabulary[end_token]]
    if end_token_positions:
        end_position = end_token_positions[-1]
    else:
        end_position = len(output_encoded)

    output_sequence = [reverse_vocabulary[val] for val in output_encoded[:end_position] if val != vocabulary[pad_token]]
    output_sentence = ''.join(output_sequence)

    return output_sentence

# Define an example sentence
sentence = "Ko te whare tenei o nga tangata katoa"

# Diacritize the sentence
diacritized_sentence = diacritize_sentence(sentence, model, vocabulary)

# Print the original and diacritized sentences
print(f"Original sentence: {sentence}")
print(f"Diacritized sentence: {diacritized_sentence}")