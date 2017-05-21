import pandas as pd
import numpy as np
import jieba
import collections
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout,Dense,Activation
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad

pos = pd.read_excel("sentiment_data/pos.xls",header=None,index_col=None)
neg = pd.read_excel("sentiment_data/neg.xls",header=None,index_col=None)
pos["mark"] = 1
neg["mark"] = 0
pn = pd.concat([pos,neg],ignore_index=True)
neglen=len(neg)
poslen=len(pos) #计算语料数目

cw = lambda x: list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)

print("word to vector")

w = []
for i in pn['words']:
    w.extend(i)

word_count = collections.Counter(w)
word_list = [x[0] for x in word_count]
maxlen = 50
# 大于50截断，小于50补0
funcw2v = lambda x:list(sequence.pad_sequences([word_list.index(word) for word in x], maxlen=maxlen))
pn["sent"] = pn["words"].apply(funcw2v)

# 奇数训练，偶数测试
x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))

print('Build model...')
model = Sequential()
model.add(Embedding(len(word_count)+1, 256))
model.add(LSTM(256, 128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

model.fit(xa, ya, batch_size=16, nb_epoch=10) #训练时间为若干个小时

print(model.evaluate(xt,yt))

# classes = model.predict_classes(xa)
# print('Test accuracy:', acc)

