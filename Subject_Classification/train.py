import pandas as pd
import os

Categories=os.listdir('dataset')
labels=[]
texts=[]

count=0
for category in Categories:
    subject_path='dataset/'+category
    subject_files=os.listdir(subject_path)
    cnt=0
    for f in subject_files:
        file=open(os.path.join(subject_path,f),'r',encoding='utf-8')
        texts.append(file.read())
        labels.append(count)
        cnt+=1
        if cnt==280:
            break
    count+=1
Data=[]
for i in range(len(labels)):
    Data.append([texts[i],labels[i]])
import random
random.shuffle(Data)
texts.clear()
labels.clear()

for text,label in Data:
    texts.append(text)
    labels.append(label)

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


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

vocab_size=4000
onehot=[one_hot(words,vocab_size) for words in corpus]

embedded_docs=pad_sequences(onehot,padding='pre',maxlen=400)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential

embedding_features=200
model=Sequential()
model.add(Embedding(vocab_size,embedding_features,input_length=400))
model.add(LSTM(100))
model.add(Dense(units=3,activation='softmax'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()

import numpy as np
y=np.array(labels)
X=np.array(embedded_docs)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=0)

from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train,num_classes=3)
y_test=to_categorical(y_test,num_classes=3)

history=model.fit(X_train, y_train, epochs=10, validation_split=0.15)
model.evaluate(X_test,y_test)
#model.save('train2.h5')

"""Time for predictions"""

para="""physical phenomena, but to model physical systems and predict how these physical systems will behave. Physicists then compare these predictions to observations or experimental evidence to show whether the theory is right or wrong.

The theories that are well supported by data and are especially simple and general are sometimes called scientific laws. Of course, all theories, including those known as laws, can be replaced by more accurate and more general laws, when a disagreement with data is found.[8]"""
corpus_t=[]    
sen=re.sub('[^a-zA-Z]',' ',para)
sen=sen.lower()
sen=sen.split()
sen=[ps.stem(word) for word in sen if word not in set(stopwords.words('english'))]
sen=' '.join(sen)
corpus_t.append(sen)
vocab_size=4000
ot=[one_hot(words,vocab_size) for words in corpus_t]

ed=pad_sequences(ot,padding='pre',maxlen=100)
X_t=np.array(ed)
pred=model.predict(X_t)
print(Categories[np.argmax(pred)])

print(pred)