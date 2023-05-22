import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping
import warnings
import os

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# Split data into training and testing sets
X_train_dev, X_test, Y_train_dev, Y_test = train_test_split(X, Y, test_size=0.2, random_state=420)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_dev, Y_train_dev, test_size=0.2, random_state=420)

# Convert Y to categorical
num_classes = len(char_to_idx)
Y_dev_cat = to_categorical(Y_dev, num_classes)
Y_train_cat = to_categorical(Y_train, num_classes)
Y_train_int = np.argmax(Y_train_cat, axis=1)
Y_dev_int = np.argmax(Y_dev_cat, axis=1)

# Define model and optimizer
def build_model(input_shape, vocab_size, batch_size, epochs, lstm_units, dense_units, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    embedding_layer = tf.keras.layers.Embedding(vocab_size, 128)(input_layer)
    lstm_layer_1 = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(embedding_layer)
    lstm_layer_2 = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(lstm_layer_1)
    flatten_layer = tf.keras.layers.Flatten()(lstm_layer_2)
    dropout_layer = tf.keras.layers.Dropout(0.5)(flatten_layer)
    dense_layer = tf.keras.layers.Dense(dense_units, activation='relu')(dropout_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dense_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# Define search space for hyperparameters
param_dist = {
    'model__batch_size': randint(32, 128),
    'model__epochs': randint(10, 50),
    'model__lstm_units': randint(64, 256),
    'model__dense_units': randint(64, 256),
    'model__learning_rate': [1e-3, 1e-4, 1e-5]
}

# Train your Keras model using fit method
model = KerasClassifier(build_fn=build_model, 
                        input_shape=(max_len,), 
                        vocab_size=len(char_to_idx),
                        batch_size=32, 
                        epochs=10, 
                        lstm_units=128, 
                        dense_units=128, 
                        learning_rate=1e-3)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Define RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    n_jobs=4,
    verbose=1,
    error_score='raise'
)

# Step 4: Train the model

random_search.fit(X_train, Y_train_int, validation_data=(X_dev, Y_dev_int))
print("Best hyperparameters: ", random_search.best_params_)
# model.fit(X, Y, batch_size=128, epochs=10, validation_split=0.2)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
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