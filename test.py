from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random
import numpy as np


f = open('./dataset.txt', 'r',  encoding='utf-8')
data = f.readlines()
random.shuffle(data)#乱序一个列表
all_data = []
labels = []
for i in range(len(data)):
    line = data[i].split('\t')
    all_data.append(line[0])
    labels.append(int(line[1].strip('\n')))
labels = np.array(labels)
# 设置最频繁使用的50000个词
MAX_NB_WORDS = 100000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 50
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 150


# 构建分词器
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# 将所有数据放到分词器里边
tokenizer.fit_on_texts(all_data)
# 文本转化为数字序列
sequences = tokenizer.texts_to_sequences(all_data)
# 构建词汇表
word_index = tokenizer.word_index

np.save('word.npy', word_index)
# 按照最大文本长度截断文本
features = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('共有 %s 个不相同的词语.' % len(word_index))
