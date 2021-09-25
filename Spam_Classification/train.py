import pandas as pd
messages=pd.read_csv('Dataset',sep='\t', names=["Label","Message"])

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

corpus=[]
for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]', ' ', messages['Message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
X=cv.fit_transform(corpus)
X=X.toarray()

y=pd.get_dummies(messages["Label"])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train,y_train)


y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

sentence="Hello, I love you my dear!"
sentence=re.sub('[^a-zA-Z]'," ",sentence)
sentence=sentence.lower()
sentence=sentence.split()
sentence=[ps.stem(word) for word in sentence if word not in stopwords.words('english')]
sentence=" ".join(sentence)
test_corpus=[]
test_corpus.append(sentence)

X_t=cv.transform(test_corpus)
X_t=X_t.toarray()
pred=model.predict(X_t)
print(pred)


print(cv.get_feature_names)






