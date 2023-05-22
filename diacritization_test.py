import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

"""def tensor_to_string(tensor):
    # converts tensor of one-hot encoded chars to string
    return ''.join([idx_to_char[idx] for idx in tensor.numpy().argmax(axis=-1)])"""

"""def vowel_accuracy(y_true, y_pred):
    y_true_str = [tensor_to_string(row) for row in y_true]
    y_pred_str = [tensor_to_string(row) for row in y_pred]
    print("y_true_str:", y_true_str[:10])
    print("y_pred_str:", y_pred_str[:10])
    vowels = {'a', 'ā', 'e', 'ē', 'i', 'ī', 'o', 'ō', 'u', 'ū'}
    correct_vowels = sum(1 for t, p in zip(y_true_str, y_pred_str) if t in vowels and t == p)
    total_vowels = sum(1 for t in y_true_str if t in vowels)
    accuracy = tf.cond(total_vowels > 0, lambda: correct_vowels / total_vowels, lambda: tf.constant(0.0))
    return accuracy"""

# Step 1: Preprocess the corpus data
with open('miwiki_data.txt', 'r', encoding='utf-8') as f:
    sentences = []
    labeled_sentences = []
    for line in f:
        line = line.strip().lower()
        # remove non-alphabetical characters and whitespace
        # line = ''.join(c for c in line if c.isalpha())
        labeled_sentences.append(line)
        # assume that diacritics are always macrons
        sentences.append(line.replace('ā', 'a').replace('ē', 'e').replace('ī', 'i').replace('ō', 'o').replace('ū', 'u'))


unique_chars = set()
for sentence in sentences:
    unique_chars.update(set(sentence))

print(sentences[0])
print(labeled_sentences[0])

# Step 2: Prepare the data for training
# create a mapping between characters and integers
char_to_idx = {char: idx for idx, char in enumerate(sorted(set(''.join(labeled_sentences))))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
# pad the sequences to a fixed length
max_len = max(len(s) for s in sentences)
X = pad_sequences(
    [[char_to_idx[c] for c in s] for s in sentences],
    maxlen=max_len,
    padding='post'
)
Y = pad_sequences(
    [[char_to_idx[c] for c in s] for s in labeled_sentences],
    maxlen=max_len,
    padding='post'
)
Y = tf.keras.utils.to_categorical(Y, num_classes=len(char_to_idx))

# Step 3: Build the model
input_shape = (max_len,)
input_layer = tf.keras.layers.Input(shape=input_shape)
embedding_layer = tf.keras.layers.Embedding(len(char_to_idx), 128)(input_layer)
lstm_layer_1 = tf.keras.layers.LSTM(128, return_sequences=True)(embedding_layer)
lstm_layer_2 = tf.keras.layers.LSTM(128, return_sequences=True)(lstm_layer_1)
dense_layer = tf.keras.layers.Dense(len(char_to_idx), activation='softmax')(lstm_layer_2)
model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)

# Step 4: Train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(X, Y, batch_size=128, epochs=10, validation_split=0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)

# Step 5: Evaluate the model
# TODO

# Step 6: Use the model to diacritize new sentences
test_sentence = 'kaore he paremata kaore he a poti he marire te tumomo'
test_sequence = np.array([[char_to_idx[c] for c in test_sentence]])
test_sequence = pad_sequences(test_sequence, maxlen=max_len, padding='post')
predicted_sequence = np.argmax(model.predict(test_sequence), axis=-1)[0]
predicted_sentence = ''.join([list(char_to_idx.keys())[i] for i in predicted_sequence])
print(predicted_sentence)

# Step 7: Save the trained model
model.save('maori_diacritization_model.h5')