import pandas as pd
data_true=pd.read_csv('True.csv')
data_fake=pd.read_csv('Fake.csv')

true_news=list(data_true['text'])[:2000]
fake_news=list(data_fake['text'])[:2000]

import random
data=[]
for text in true_news:
    data.append([text,1])
for text in fake_news:
    data.append([text,0])
    
random.shuffle(data)

y=[]
for i in range(len(data)):
    y.append(data[i][1])

texts=[]
for sen in data:
    texts.append(sen[0])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
lem=WordNetLemmatizer()
corpus=[]
for i in range(len(texts)):
    sen=re.sub('[^a-zA-Z]', ' ', texts[i])
    sen=sen.lower()
    sen=sen.split()
    sen=[ps.stem(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    corpus.append(sen)

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

vocab_size=2000

onehot=[one_hot(words,vocab_size) for words in corpus]
sent_length=20
embedded_docs=pad_sequences(onehot,padding='pre',maxlen=sent_length)

embedding_vector_features=40
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])

model.summary()
import numpy as np

X=np.array(embedded_docs)
y=np.array(y)
print(X.shape,y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=0)

model.fit(X_train, y_train, epochs=10, validation_split=0.2)

y_pred=model.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i]>=0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

model.save('train_model1.h5')


