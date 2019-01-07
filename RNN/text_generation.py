from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
import numpy as np

# read data
lines = []
with open("data/alice.txt", 'r') as fin:
    for line in fin.readlines():
        line = line.strip().lower()
        if len(line) == 0:
            continue
        lines.append(line)
text = " ".join(lines)

# build dict
chars = set([c for c in text])
len_chars = len(chars)
chars2Index = dict((c, i) for i, c in enumerate(chars))
index2Char = dict((i, c) for i, c in enumerate(chars))

# build train set
len_seq = 10
step = 1
input_chars = []
label_chars = []
for i in range(0, len(text) - len_seq, step):
    input_chars.append(text[i:i + len_seq])
    label_chars.append(text[i + len_seq])

# token train set
X = np.zeros((len(input_chars), len_seq, len_chars), dtype=np.bool)
y = np.zeros((len(input_chars), len_chars))
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, chars2Index[ch]] = 1
    y[i, chars2Index[label_chars[i]]] = 1

print(X.shape)
print(y.shape)

# build model
hidden_size = 256  # inner output size, viewed as h; more, the more corpus needed, less, the repeated words occur
batch_size = 128
num_iterations = 25
num_epochs_per_iteration = 1
num_preds_per_epoch = 100  # predict continuously

model = Sequential()
model.add(SimpleRNN(hidden_size, return_sequences=False, input_shape=(len_seq, len_chars), unroll=True))
model.add(Dense(len_chars))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer='rmsprop')
model.summary()

for iteration in range(num_iterations):
    print("=" * 50)
    print("Iteration #: %d" % iteration)

    model.fit(X, y, batch_size=batch_size, epochs=num_epochs_per_iteration)

    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating from seed: %s" % test_chars)
    print(test_chars, end="")
    for i in range(num_preds_per_epoch):
        X_test = np.zeros((1, len_seq, len_chars))
        for j, ch in enumerate(test_chars):
            X_test[0, j, chars2Index[ch]] = 1
        pred = model.predict(X_test)[0]
        y_pred = index2Char[np.argmax(pred)]
        print(y_pred, end="")
        test_chars = test_chars[1:] + y_pred
    print()
