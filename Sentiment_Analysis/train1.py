import pandas as pd
data=pd.read_csv('Dataset.tsv',sep='\t')
texts=list(data['review'])[:5000]
labels=list(data['sentiment'])[:5000]

for i in range(len(texts)):
    texts[i]=texts[i][:300]

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
corpus=[]
for i in range(len(texts)):
    sen=re.sub('[^a-zA-Z]',' ',texts[i])
    sen=sen.lower()
    sen=sen.split()
    sen=[ps.stem(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    corpus.append(sen)
max1=100000
for i in range(len(corpus)):
    if len(corpus[i])<max1:
        max1=len(corpus[i])

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

vocab_size=4000
onehot=[one_hot(words,vocab_size) for words in corpus]

embedded_docs=pad_sequences(onehot,padding='pre',maxlen=300)

import numpy as np
X=np.array(embedded_docs)
y=np.array(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential

embedding_features=200
model=Sequential()
model.add(Embedding(vocab_size,embedding_features,input_length=300))
model.add(LSTM(200))
model.add(Dense(units=2, activation='softmax'))

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()

history=model.fit(X_train,y_train,epochs=10,validation_split=0.2)
model.evaluate(X_test,y_test)
model.save('train.h5')