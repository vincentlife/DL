from keras.models import Sequential
model = Sequential()
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout,Dense,Activation
from keras.utils import plot_model
from keras import backend as K


model = Sequential()
model.add(Embedding(400, 256))
model.add(LSTM(units=256)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(units=128))
model.add(Activation('sigmoid'))
with K.get_session():
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
    model.summary()