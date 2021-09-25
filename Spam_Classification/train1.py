import pandas as pd
dataset=pd.read_csv('Dataset', sep='\t', names=["Label", "Message"])

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

ps=PorterStemmer()
lm=WordNetLemmatizer()
corpus=[]
for i in range(len(dataset)):
    sen=re.sub('[^a-zA-Z]',' ', dataset["Message"][i])
    sen=sen.lower()
    sen=sen.split()
    sen=[lm.lemmatize(word) for word in sen if word not in set(stopwords.words('english'))]
    sen=' '.join(sen)
    corpus.append(sen)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
X=cv.fit_transform(corpus)

X=X.toarray()

y=pd.get_dummies(dataset["Label"])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

sentence="""IMPORTANT INFORMATION:

The new domain names are finally available to the general public at discount prices. Now you can register one of the exciting new .BIZ or .INFO domain names, as well as the original .COM and .NET names for just $14.95. These brand new domain extensions were recently approved by ICANN and have the same rights as the original .COM and .NET domain names. The biggest benefit is of-course that the .BIZ and .INFO domain names are currently more available. i.e. it will be much easier to register an attractive and easy-to-remember domain name for the same price.  Visit: http://www.affordable-domains.com today for more info."""
sentence=re.sub("[^a-zA-Z]"," ",sentence)
sentence=sentence.lower()
sentence=sentence.split()
sentence=[ps.stem(word) for word in sentence if word not in stopwords.words('english')]
sentence=' '.join(sentence)

corpust=[]
corpust.append(sentence)
X_t=cv.transform(corpust)
X_t=X_t.toarray()



predictions=model.predict(X_t)

