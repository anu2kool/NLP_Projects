import pandas as pd
data=pd.read_csv('Dataset',sep='\t',names=['Label','Text'])
texts=list(data['Text'])
labels=list(data['Label'])

for i in range(len(labels)):
    if labels[i]=='ham':
        labels[i]=1
    else:
        labels[i]=0


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

corpus=[]
ps=PorterStemmer()
for i in range(len(texts)):
    sen=re.sub('[^a-zA-Z]',' ',texts[i])
    sen=sen.lower()
    sen=sen.split()
    sen=[ps.stem(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    corpus.append(sen)

import tensorflow
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

vocab_size=2000
onehot=[one_hot(words,vocab_size) for words in corpus]

embedded_docs=pad_sequences(onehot,padding='pre',maxlen=20)
embedding_features=40
model=Sequential()
model.add(Embedding(vocab_size,embedding_features,input_length=20))
model.add(LSTM(100))
model.add(Dense(units=2,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

import numpy as np
X=np.array(embedded_docs)
y=np.array(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

history=model.fit(X_train,y_train,epochs=10,validation_split=0.2)

model.evaluate(X_test,y_test)
y_pred=model.predict(X_test)

new_y_test=[]
new_y_pred=[]

for i in range(len(y_test)):
    new_y_test.append(np.argmax(y_test,axis=1)[i])
    new_y_pred.append(np.argmax(y_pred,axis=1)[i])
    

new_y_pred=np.array(new_y_pred)
new_y_test=np.array(new_y_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(new_y_test,new_y_pred)
accuracy=accuracy_score(new_y_test,new_y_pred)

model.save('Train1.h5')
