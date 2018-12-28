from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam


class LeNet:
    NUM_CLASS = 10
    IMG_ROWS, IMG_COLS, NUM_CHANNELS = 32, 32, 3
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, NUM_CHANNELS)

    EPOCH = 20
    BATCH_SIZE = 128
    VERBOSE = 2
    DROPOUT = 0.3
    VALIDATION_SPLIT = 0.2

    OPTIMIZER = Adam()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=0,
        mode='auto'
    )
    checkpoint = ModelCheckpoint(filepath='model/CIFAR-10.model', save_best_only=True)
    # tensorboard --logdir=CNN/log/
    tensorboard = TensorBoard(log_dir='log/', write_graph=False)

    model = None

    @classmethod
    def build(cls):
        cls.model = Sequential()

        cls.model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=cls.INPUT_SHAPE))
        cls.model.add(Activation('relu'))
        cls.model.add(MaxPooling2D(pool_size=(2, 2)))  # default strides=pool_size

        cls.model.add(Dropout(cls.DROPOUT))

        cls.model.add(Flatten())

        cls.model.add(Dense(512))
        cls.model.add(Activation('relu'))

        cls.model.add(Dropout(cls.DROPOUT))

        cls.model.add(Dense(cls.NUM_CLASS))
        cls.model.add(Activation("softmax"))

        cls.model.summary()
        cls.model.compile(loss='categorical_crossentropy', optimizer=cls.OPTIMIZER, metrics=['accuracy'])

    @classmethod
    def get_data(cls):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.reshape(-1, cls.IMG_ROWS, cls.IMG_COLS, cls.NUM_CHANNELS) / 255.0
        X_test = X_test.reshape(-1, cls.IMG_ROWS, cls.IMG_COLS, cls.NUM_CHANNELS) / 255.0

        y_train = np_utils.to_categorical(y_train, cls.NUM_CLASS)
        y_test = np_utils.to_categorical(y_test, cls.NUM_CLASS)

        return (X_train, y_train), (X_test, y_test)

    @classmethod
    def train(cls, X_train, y_train):
        cls.model.fit(X_train, y_train, batch_size=cls.BATCH_SIZE, epochs=cls.EPOCH, verbose=cls.VERBOSE,
                      validation_split=cls.VALIDATION_SPLIT,
                      callbacks=[cls.early_stopping, cls.checkpoint, cls.tensorboard])

    @classmethod
    def evaluate(cls, X_test, y_test):
        score = cls.model.evaluate(X_test, y_test, verbose=cls.VERBOSE)
        print("loss:", score[0])
        print("accuracy:", score[1])


LeNet.build()
(X_train, y_train), (X_test, y_test) = LeNet.get_data()
LeNet.train(X_train, y_train)
LeNet.evaluate(X_test, y_test)
