import pandas as pd
import os

Categories=os.listdir('dataset')
labels=[]
texts=[]

count=0

for category in Categories:
    subject_path='dataset/'+category
    subject_files=os.listdir(subject_path)
    for f in subject_files:
        file=open(os.path.join(subject_path,f),'r',encoding='utf-8')
        texts.append(file.read())
        labels.append(count)
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
lem=WordNetLemmatizer()
for i in range(len(texts)):
    sen=re.sub('[^a-zA-Z]',' ',texts[i])
    sen=sen.lower()
    sen=sen.split()
    sen=[lem.lemmatize(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    corpus.append(sen)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

vocab_size=10000
onehot=[one_hot(words,vocab_size) for words in corpus]

embedded_docs=pad_sequences(onehot,padding='pre',maxlen=1000)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential


embedding_features=1000
model=Sequential()
model.add(Embedding(vocab_size,embedding_features,input_length=2000))
model.add(LSTM(100))
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()

import numpy as np
y=np.array(labels)
X=np.array(embedded_docs)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=0)

from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

history=model.fit(X_train, y_train, epochs=10, validation_split=0.15)
model.evaluate(X_test,y_test)

#model.save('trainlemmatizer2.h5')

"""Time for predictions"""

para="""Do you have a feeling you might be an entrepreneur at heart? In “50 Signs You Might Be an Entrepreneur,” published on Entrepreneur.com, John Rampton could point out the one—or many—things that makes you the perfect small business owner deep down inside.

Entrepreneurs and business owners have a certain kind of spirit and drive that keeps pushing them forward. Use this business article to find out if you possess the qualities of an entrepreneur yourself"""
corpus_t=[]    
sen=re.sub('[^a-zA-Z]',' ',para)
sen=sen.lower()
sen=sen.split()
sen=[lem.lemmatize(word) for word in sen if word not in set(stopwords.words('english'))]
sen=' '.join(sen)
corpus_t.append(sen)
vocab_size=4000
ot=[one_hot(words,vocab_size) for words in corpus_t]

ed=pad_sequences(ot,padding='pre',maxlen=400)
X_t=np.array(ed)
pred=model.predict(X_t)
print(Categories[np.argmax(pred)])
print(np.max(pred)*100)

print(pred)
