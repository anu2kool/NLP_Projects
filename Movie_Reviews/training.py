import pandas as pd
data=pd.read_csv('Train.csv')

texts=list(data['text'])[:6000]
labels=list(data['label'])[:6000]


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
corpus=[]
for i in range(len(texts)):
    sen=re.sub('[^a-zA-Z]', ' ', texts[i])
    sen=sen.lower()
    sen=sen.split()
    sen=[ps.stem(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    corpus.append(sen)

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus)
X=X.toarray()

import numpy as np
y=np.array(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.naive_bayes import MultinomialNB
model=SGDClassifier().fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

from sklearn.externals import joblib
joblib.dump(model,'Train_Model1.pkl')
joblib.dump(cv.vocabulary_,'vectorizer.pkl')
print(cv.vocabulary_)
