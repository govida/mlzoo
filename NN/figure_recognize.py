import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import regularizers

np.random.seed(10086)

EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 2
NUM_CLASS = 10
OPTIMIZER = Adam()
HIDDEN = 128
DROPOUT = 0.3
VALIDATION_SPLIT = 0.2

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=0,
    mode='auto'
)
checkpoint = ModelCheckpoint(filepath='model/figure_recognize.model', save_best_only=True)
# tensorboard --logdir=NN/log/logs/
tensorboard = TensorBoard(log_dir='log/', histogram_freq=0, write_graph=True, write_images=False)

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
(X_train, y_train), (X_test, y_test) = mnist.load_data()

RESHAPE = 784
X_train = X_train.reshape(-1, RESHAPE) / 255.0
X_test = X_test.reshape(-1, RESHAPE) / 255.0

y_train = np_utils.to_categorical(y_train, NUM_CLASS)
y_test = np_utils.to_categorical(y_test, NUM_CLASS)

model = Sequential()
model.add(Dense(HIDDEN, input_shape=(RESHAPE,)))
model.add(Activation("relu"))
model.add(Dense(HIDDEN, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(NUM_CLASS))
model.add(Activation("softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT, callbacks=[early_stopping, checkpoint, tensorboard])
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("loss:", score[0])
print("accuracy:", score[1])
