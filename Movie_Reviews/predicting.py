from sklearn.externals import joblib
model=joblib.load('Train_Model1.pkl')
vocab=joblib.load('vectorizer.pkl')
print(vocab)
para="""
A negative sentence is a sentence that states that something is false. In English, we create negative sentences by adding the word 'not' after the auxiliary, or helping, verb.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

corpus_t=[]
para=re.sub('[^a-zA-Z]',' ',para)
para=para.lower()
para=para.split()
para=[ps.stem(word) for word in para if word not in set(stopwords.words('english'))]
para=' '.join(para)
corpus_t.append(para)

from sklearn.feature_extraction.text import CountVectorizer
cv1=CountVectorizer(vocabulary=vocab)

X_t=cv1.transform(corpus_t)
X_t=X_t.toarray()

pred=model.predict(X_t)
