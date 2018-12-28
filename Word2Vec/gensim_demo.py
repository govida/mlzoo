from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Text8Sentences(object):

    def __init__(self, fpath, maxlen):
        self.fpath = fpath
        self.maxlen = maxlen

    def __iter__(self):
        with open(self.fpath, "r") as ftext:
            text = ftext.read().split()
            words = []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                words.append(word)
            yield words


sentences = Text8Sentences('/Users/davidsheng/PycharmProjects/mlzoo/Word2Vec/data/text8', 50)
model = word2vec.Word2Vec(sentences, size=100, min_count=30)

print("""model.most_similar("woman")""")
print(model.most_similar("woman"))

print("""model.most_similar(positive=["woman", "king"], negative=["man"], topn=10)""")
print(model.most_similar(positive=['woman', 'king'],
                         negative=['man'],
                         topn=10))

print("""model.similarity("girl", "woman")""")
print(model.similarity("girl", "woman"))
print("""model.similarity("girl", "man")""")
print(model.similarity("girl", "man"))
print("""model.similarity("girl", "car")""")
print(model.similarity("girl", "car"))
print("""model.similarity("bus", "car")""")
print(model.similarity("bus", "car"))

# 实际操作中对权重的进一步处理
# 第一行 补0 embedding
# 最后一行 为 缺失值

import numpy as np
import pickle

weights = model.wv.syn0
vocab = dict([(k, v.index + 1) for k, v in model.wv.vocab.items()])

# 注意 这里是 +2 （补0 mask，缺失值）
embed_weights = np.zeros(shape=(weights.shape[0] + 2, weights.shape[1]))
# 中间权重
embed_weights[1:weights.shape[0] + 1] = weights
# 缺失值
unk_vec = np.random.random(size=weights.shape[1]) * 0.5
embed_weights[weights.shape[0] + 1] = unk_vec - unk_vec.mean()

pickle.dump(vocab, open("data/word_embed_.dict.pkl", "wb"))
np.save("data/word_embed_.npy", embed_weights)
