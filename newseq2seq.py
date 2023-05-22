import tensorflow as tf
import numpy as np
import re
import unicodedata

# Set parameters
BATCH_SIZE = 64
LATENT_DIM = 256
EPOCHS = 100
MAX_WORD_LEN = 30
MAX_SENTENCE_LEN = 160

# Load dataset
with open('miwiki_data.txt', 'r') as f:
    sentences = f.read().splitlines()

# Define char vocabulary
char_vocab = set()
for sentence in sentences:
    char_vocab.update(set(sentence))
char_vocab = sorted(list(char_vocab))
char_vocab_size = len(char_vocab) + 2

# Define char-to-index and index-to-char mappings
char_to_index = {char: i for i, char in enumerate(char_vocab)}
char_to_index['<start>'] = len(char_to_index)
char_to_index['<end>'] = len(char_to_index)
index_to_char = {i: char for i, char in enumerate(char_vocab)}

# Preprocess dataset
def encode_sentence(sentence):
    encoded_sentence = []
    for word in sentence.split():
        consonants = re.findall('[^aeiouāēīōū]', word)
        vowels = re.findall('[aeiouāēīōū]', word)
        encoded_consonants = [char_to_index[char] for char in consonants]
        encoded_vowels = [char_to_index[char] for char in vowels]
        # Pad sequences of consonants and vowels separately
        padded_consonants = tf.keras.preprocessing.sequence.pad_sequences([encoded_consonants], maxlen=MAX_WORD_LEN, padding='post', truncating='post')[0]
        padded_vowels = tf.keras.preprocessing.sequence.pad_sequences([encoded_vowels], maxlen=MAX_WORD_LEN, padding='post', truncating='post')[0]
        # Pad the shorter sequence with zeros to make the concatenated word the same length
        if len(padded_consonants) > len(padded_vowels):
            padded_vowels = np.pad(padded_vowels, (0, len(padded_consonants) - len(padded_vowels)), mode='constant', constant_values=0)
        else:
            padded_consonants = np.pad(padded_consonants, (0, len(padded_vowels) - len(padded_consonants)), mode='constant', constant_values=0)
        int_consonants = np.array([int(i) for i in padded_consonants])
        int_vowels = np.array([int(i) for i in padded_vowels])
        encoded_word = np.concatenate((int_consonants, int_vowels), axis=0)
        encoded_sentence.append(encoded_word)
    # Pad sequences of words
    padded_sentence = tf.keras.preprocessing.sequence.pad_sequences([encoded_sentence], maxlen=MAX_SENTENCE_LEN*2, padding='post', truncating='post')[0]
    padded_sentence = padded_sentence.reshape(-1, 2)
    # Compute length of encoded sentence
    length = min(len(sentence), MAX_SENTENCE_LEN) * 2
    return padded_sentence, length

input_data = []
output_data = []
encoder_input_lengths = []
decoder_input_data = []
decoder_output_data = []
max_encoder_input_len = 0
max_decoder_input_len = 0
max_decoder_output_len = 0

unique_sentences = set(sentences)
filtered_sentences = [sentence for sentence in unique_sentences if len(sentence.split()) > 2]

for sentence in filtered_sentences:
    # Remove diacritics from input
    input_sentence = unicodedata.normalize('NFKD', sentence).encode('ASCII', 'ignore').decode('utf-8')
    # Keep original sentence for output
    output_sentence = sentence
    # Encode input and output sentences
    encoded_input, input_length = encode_sentence(input_sentence)
    encoded_output, output_length = encode_sentence(output_sentence)
    input_data.append(encoded_input) # append encoded sentence to input_data
    output_data.append(encoded_output)
    # Add start and end tokens to decoder input and output
    decoder_input = [char_to_index['<start>']] + encoded_output[:-1, 0].tolist() + [char_to_index['<end>']]
    decoder_output = encoded_output[1:, 1]
    # Pad decoder input and output data
    padded_decoder_input = np.pad(decoder_input, (0, MAX_SENTENCE_LEN*2 - output_length + 2), mode='constant', constant_values=0)
    padded_decoder_output = np.pad(decoder_output, (0, MAX_SENTENCE_LEN*2 - output_length + 1), mode='constant', constant_values=char_to_index['<end>'])
    padded_decoder_input = np.array(padded_decoder_input)
    max_decoder_input_len = max(max_decoder_input_len, len(padded_decoder_input))
    padded_decoder_output = np.array(padded_decoder_output)
    max_decoder_output_len = max(max_decoder_output_len, len(padded_decoder_output))
    padded_decoder_output = tf.keras.utils.to_categorical(padded_decoder_output, num_classes=char_vocab_size)
    decoder_input_data.append(padded_decoder_input)
    decoder_output_data.append(padded_decoder_output)

# Pad sequences of words for encoder input
padded_input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=MAX_SENTENCE_LEN*2, padding='post', truncating='post')

# Pad decoder input and output data to have fixed length
decoder_input_data = np.array([np.pad(d, (0, max_decoder_input_len - len(d)), mode='constant', constant_values=0) for d in decoder_input_data])
decoder_output_data = np.array([np.pad(d, ((0, max_decoder_output_len - d.shape[0]), (0, 0)), mode='constant', constant_values=0) for d in decoder_output_data])

# Convert data to numpy arrays
encoder_input_data = np.array(padded_input_data)
decoder_input_data = np.array(decoder_input_data)
decoder_output_data = np.array(decoder_output_data)

# Define encoder model
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(char_vocab_size, LATENT_DIM)
encoder_lstm = tf.keras.layers.LSTM(LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [state_h, state_c]
encoder_model = tf.keras.models.Model(inputs=encoder_inputs, outputs=encoder_states)

# Define decoder
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(char_vocab_size, LATENT_DIM)
decoder_lstm = tf.keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(char_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define encoder inference model
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

# Define decoder inference model
decoder_state_input_h = tf.keras.layers.Input(shape=(LATENT_DIM,))
decoder_state_input_c = tf.keras.layers.Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Define the complete model
model_inputs = encoder_inputs + [decoder_inputs]
model_outputs = decoder_outputs
model = tf.keras.models.Model(inputs=model_inputs, outputs=model_outputs)

model.summary()

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.CategoricalCrossentropy()

# Define function to calculate loss
def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_function(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function)

# Train the model
for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
    for i in range(0, len(encoder_input_data), BATCH_SIZE):
        # Get batch
        encoder_input_batch = encoder_input_data[i:i+BATCH_SIZE]
        decoder_input_batch = decoder_input_data[i:i+BATCH_SIZE]
        decoder_output_batch = decoder_output_data[i:i+BATCH_SIZE]

        # Compute gradients and update weights
        with tf.GradientTape() as tape:
            logits = model([encoder_input_batch, decoder_input_batch])
            loss_value = loss_function(decoder_output_batch, logits)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Print progress
        print('\rBatch {}/{} Loss {:.4f}'.format(i + 1, len(encoder_input_data), loss_value), end='')
    print()

# Save the model
model.save('diacritizer.h5')

# Define function to diacritize sentence
def diacritize_sentence(sentence):
    # Encode input sentence
    input_data, _ = encode_sentence(sentence)
    # Get encoder states
    states_value = encoder_model.predict(input_data)
    state_h_new = np.zeros((1, 256))
    state_h_new[0,:] = states_value[0][0,:]
    state_c_new = np.zeros((1, 256))
    state_c_new[0,:] = states_value[1][0,:]
    # Generate target sequence of length 1
    target_seq = np.zeros((1, 1))
    # Set the first character of target sequence to the start token
    target_seq[0, 0] = char_to_index['<start>']
    # Initialize context with zeros
    context = np.zeros((1, 2*MAX_WORD_LEN))
    # Generate diacritized sentence
    diacritized_sentence = ''
    while True:
        # Concatenate encoder output and context with target sequence
        target_data = np.concatenate([context, target_seq], axis=-1).reshape((1, 1, 2*MAX_WORD_LEN+1))
        # Get decoder outputs and states
        decoder_outputs, state_h, state_c = decoder_model.predict([target_data] + [s for s in states_value])
        # Get character index with highest probability
        char_index = np.argmax(decoder_outputs[0, -1, :])
        # Get character from index
        char = index_to_char[char_index]
        # Exit loop if end token is reached or max sentence length is exceeded
        if char == '<end>' or len(diacritized_sentence) >= MAX_SENTENCE_LEN:
            break
        # Add diacritic to vowels and add character to diacritized sentence
        if char in ['a', 'e', 'i', 'o', 'u', 'ā', 'ē', 'ī', 'ō', 'ū']:
            diacritic = index_to_char[np.argmax(decoder_outputs[0, -2, :])]
            char = diacritic + char
        diacritized_sentence += char
        # Update context with predicted diacritic
        if char in ['a', 'e', 'i', 'o', 'u', 'ā', 'ē', 'ī', 'ō', 'ū']:
            context[0, MAX_WORD_LEN + len(diacritized_sentence) - 2] = char_to_index[char]
        # Update target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = char_to_index[char]
        # Update decoder states
        states_value = [state_h, state_c]
    return diacritized_sentence

# Test diacritize_sentence function
test_sentence = 'kia ora te wiki o te reo māori'
diacritized_test_sentence = diacritize_sentence(test_sentence)
print('Input sentence:', test_sentence)
print('Diacritized sentence:', diacritized_test_sentence)