from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import skipgrams

text = 'I love green eggs and ham.'
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}
print(word2id)
print(id2word)

wids = [word2id[w] for w in text_to_word_sequence(text)]
print(wids)
pairs, labels = skipgrams(wids, len(word2id), window_size=2)
for pair, label in zip(pairs, labels):
    print(pair, label)
