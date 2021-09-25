import numpy as np
import pandas as pd

fake_news=pd.read_csv('Fake.csv')
true_news=pd.read_csv('True.csv')

fake_text=list(fake_news['text'])
true_text=list(true_news['text'])

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer

texts=[]
ps=PorterStemmer()
lem=WordNetLemmatizer()
count=0
for i in range(len(fake_text)):
    sen=re.sub('[^a-zA-Z]',' ',fake_text[i][:200])
    sen=sen.lower()
    sen=sen.split()
    sen=[lem.lemmatize(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    texts.append([sen,0])
    count+=1
    if count==10000:
        break
count=0
for i in range(len(true_text)):
    sen=re.sub('[^a-zA-Z]',' ',true_text[i][:200])
    sen=sen.lower()
    sen=sen.split()
    sen=[lem.lemmatize(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    texts.append([sen,1])
    count+=1
    if count==10000:
        break

import random
random.shuffle(texts)
corpus=[]
y=[]
for sen,label in texts:
    corpus.append(sen)
    y.append(label)

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X=cv.fit_transform(corpus)
X=X.toarray()

y=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

text="""
Facebook has submitted before the high court that it cannot remove any allegedly illegal group, like the bois locker room, from its platform as removal of such accounts or blocking access to them came under the purview of the discretionary powers of the government according to the Information Technology (IT) Act
"""

corp=[]
text=re.sub('[^a-zA-Z]'," ",text)
text=text.lower()
text=text.split()
text=[lem.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
text=' '.join(text)
corp.append(text)
X_t=cv.transform(corp)
X_t=X_t.toarray()

pred=model.predict(X_t)
