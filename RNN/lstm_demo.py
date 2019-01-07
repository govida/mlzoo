'''
    UMICH SI650
    https://www.kaggle.com/c/si650winter11
'''
import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, SpatialDropout1D, GRU,LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

np.random.seed(42)

# generate data
word_set = set()
max_len = 0
with open('data/UMICH SI650/training.txt', 'r') as f:
    for line in f:
        _, sentence = line.split('\t')
        words = [x.lower() for x in nltk.word_tokenize(sentence)]
        if len(words) > max_len:
            max_len = len(words)
        for word in words:
            word_set.add(word)

word2index = {}
index2word = {}
for i, word in enumerate(word_set):
    word2index[word] = i
    index2word[i] = word

X = []
y = []
with open('data/UMICH SI650/training.txt', 'r') as f:
    for line in f:
        label, sentence = line.split('\t')
        words = [x.lower() for x in nltk.word_tokenize(sentence)]
        word_indexes = [word2index[word] for word in words]
        X.append(word_indexes)
        y.append(label)

X = pad_sequences(X, max_len, padding='post')
y = np_utils.to_categorical(y)

# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = Sequential()
model.add(Embedding(len(word_set), 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
# LSTM 可以用 GRU 替换，最大区别 GRU 训练快
model.add(LSTM(64,recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=0,
    mode='auto'
)
checkpoint = ModelCheckpoint(filepath='model/lstm.model', save_best_only=True)
# tensorboard --logdir=*/log/
tensorboard = TensorBoard(log_dir='log/', write_graph=False)

model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_test, y_test),
          callbacks=[early_stopping, checkpoint, tensorboard])