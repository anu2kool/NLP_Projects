import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy

nlp=spacy.load('en_core_web_sm')


stopwords=['very','various', 'often', 'own', 'already', 'whom', 'our', 'whereas', "'re", 'who', 'during', 'so', 'him', 'five', 'an', 'since', 'whereafter', 'twenty', 'amount', 'some', 'eleven', 'itself', 'here', 'would', 'amongst', 'others', 'thereupon', 'elsewhere', 'becoming', 'is', 'should', 'are', 'do', 'very', 'always', 'toward', 'throughout', 'hereby', 'never', "'ll", 'seemed', 'that', 'doing', 'wherever', 'give', 'over', 'which', 'one', 'regarding', 'were', 'whose', "n't", 'the', 'then', 'get', 'became', 'they', 'front', 'top', 'go', 'using', 'beyond', 'for', 'ca', 'ten', 'whither', 'still', 'around', '‘m', 'or', 'perhaps', 'how', 'back', 'otherwise', 'seeming', 'eight', 'to', 'at', 'yet', 'again', 'see', 'has', 'me', 'beforehand', 'please', 'put', 'per', 'my', 'part', 'n’t', 'ever', 'sometime', 'if', 'now', 'through', 'side', 'she', 'am', 'several', "'d", 'both', 'but', 'former', 'become', 'say', 'whether', 'whereupon', 'anyway', 'among', 'empty', 'something', 'hereafter', 'namely', 'n‘t', 'yourself', 'must', 'hers', 'other', 'make', 'four', 'three', 'alone', 'did', 'meanwhile', 'due', 'where', 'we', 'becomes', '’s', 'above', 'everyone', '’ve', 'therein', 'whole', 'hence', 'nobody', 'i', 'fifteen', 'though', 'together', 'all', 'mine', 'thereafter', 'thence', "'m", 'just', 'nine', 'along', '‘re', 'sometimes', 'unless', "'s", 'two', 'it', '‘d', 'noone', 'another', 'really', 'these', 'made', 'may', 'her', 'and', 'their', 'via', 'anywhere', 'full', 'bottom', 'move', 'hundred', 'whoever', 'afterwards', 'does', 'of', 'name', 'behind', 'third', 'between', 'across', 'someone', 'also', 'latterly', 'enough', 'than', 'seem', 'there', 'your', 'within', 'formerly', 'thru', 'this', 'whenever', 'forty', 'show', 'into', 'with', 'except', 'about', 'therefore', '’re', 'why', 'either', 'after', '‘s', 'same', 'anything', 'fifty', 'from', 'sixty', 'keep', 'only', 'besides', 'a', 'beside', 'any', 'out', 'done', 'when', 'more', 'call', 'least', 'been', 'onto', 'else', 'next', "'ve", 'latter', 'us', 'whereby', 'everything', 'further', 'herself', '’ll', 'anyone', 'indeed', 'each', 'under', 'yourselves', 'on', 'themselves', 'take', 'had', 'he', 'twelve', 'nowhere', 'what', 'last', 'ourselves', 'while', 'off', 'be', 'hereupon', 'too', 'somehow', 'was', 'first', 'have', 'by', 'ours', 'rather', 'serious', 'will', 'nevertheless', 'upon', '‘ll', 'wherein', 'myself', 'moreover', 'however', 'until', 'herein', 'whatever', 'might', '‘ve', 'as', 'them', 'once', 'seems', 'used', 'whence', 'somewhere', 'in', 'towards', 'its', 'because', 'himself', 'down', 'those', 'up', 'his', 'thus', 'less', '’m', 'every', 'anyhow', 'although', 'everywhere', 'such', 'even', 'almost', '’d', 'many', 're', 'six', 'before', 'being', 'few', 'thereby', 'yours', 'you', 'quite']

doc=nlp('I am playing with my new doll')
for token in doc:
    print(token.text,token.pos_)

import pandas as pd
data_amazon=pd.read_csv('amazon_dataset.txt',sep='\t',header=None,names=['review','sentiment'])
data_amazon.head()

data_imdb=pd.read_csv('imdb_dataset.txt',sep='\t',header=None,names=['review','sentiment'])
data_imdb.head()

data_yelp=pd.read_csv('yelp_dataset.txt',sep='\t',header=None,names=['review','sentiment'])
data_yelp.head()

data=data_yelp.append([data_amazon,data_imdb],ignore_index=True)
print(data.shape)

import string
def clean_text(sentence):
    doc=nlp(sentence)
    tokens=[]
    for token in doc:
        if token.lemma_!='-PRON-':
            temp=token.lemma_.lower().strip()
        else:
            temp=token.lower_
        tokens.append(temp)
    clean_tokens=[]
    for token in tokens:
        if token not in stopwords and token not in punct:
            clean_tokens.append(token)
    
    return clean_tokens


punct=string.punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

tfidf=TfidfVectorizer(tokenizer=clean_text)
classifier=MultinomialNB()

X=data['review']
y=data['sentiment']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

clf=Pipeline([('tfidf',tfidf), ('clf',classifier)])

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
print(report)








print(stopwords)

stopwords=['various', 'often', 'own', 'already', 'whom', 'our', 'whereas', "'re", 'who', 'during', 'so', 'him', 'five', 'an', 'since', 'whereafter', 'twenty', 'amount', 'some', 'eleven', 'itself', 'here', 'would', 'amongst', 'others', 'thereupon', 'elsewhere', 'becoming', 'is', 'should', 'are', 'do', 'very', 'always', 'toward', 'throughout', 'hereby', 'never', "'ll", 'seemed', 'that', 'doing', 'wherever', 'give', 'over', 'which', 'one', 'regarding', 'were', 'whose', "n't", 'the', 'then', 'get', 'became', 'they', 'front', 'top', 'go', 'using', 'beyond', 'for', 'ca', 'ten', 'whither', 'still', 'around', '‘m', 'or', 'perhaps', 'how', 'back', 'otherwise', 'seeming', 'eight', 'to', 'at', 'yet', 'again', 'see', 'has', 'me', 'beforehand', 'please', 'put', 'per', 'my', 'part', 'n’t', 'ever', 'sometime', 'if', 'now', 'through', 'side', 'she', 'am', 'several', "'d", 'both', 'but', 'former', 'become', 'say', 'whether', 'whereupon', 'anyway', 'among', 'empty', 'something', 'hereafter', 'namely', 'n‘t', 'yourself', 'must', 'hers', 'other', 'make', 'four', 'three', 'alone', 'did', 'meanwhile', 'due', 'where', 'we', 'becomes', '’s', 'above', 'everyone', '’ve', 'therein', 'whole', 'hence', 'nobody', 'i', 'fifteen', 'though', 'together', 'all', 'mine', 'thereafter', 'thence', "'m", 'just', 'nine', 'along', '‘re', 'sometimes', 'unless', "'s", 'two', 'it', '‘d', 'noone', 'another', 'really', 'these', 'made', 'may', 'her', 'and', 'their', 'via', 'anywhere', 'full', 'bottom', 'move', 'hundred', 'whoever', 'afterwards', 'does', 'of', 'name', 'behind', 'third', 'between', 'across', 'someone', 'also', 'latterly', 'enough', 'than', 'seem', 'there', 'your', 'within', 'formerly', 'thru', 'this', 'whenever', 'forty', 'show', 'into', 'with', 'except', 'about', 'therefore', '’re', 'why', 'either', 'after', '‘s', 'same', 'anything', 'fifty', 'from', 'sixty', 'keep', 'only', 'besides', 'a', 'beside', 'any', 'out', 'done', 'when', 'more', 'call', 'least', 'been', 'onto', 'else', 'next', "'ve", 'latter', 'us', 'whereby', 'everything', 'further', 'herself', '’ll', 'anyone', 'indeed', 'each', 'under', 'yourselves', 'on', 'themselves', 'take', 'had', 'he', 'twelve', 'nowhere', 'what', 'last', 'ourselves', 'while', 'off', 'be', 'hereupon', 'too', 'somehow', 'was', 'first', 'have', 'by', 'ours', 'rather', 'serious', 'will', 'nevertheless', 'upon', '‘ll', 'wherein', 'myself', 'moreover', 'however', 'until', 'herein', 'whatever', 'might', '‘ve', 'as', 'them', 'once', 'seems', 'used', 'whence', 'somewhere', 'in', 'towards', 'its', 'because', 'himself', 'down', 'those', 'up', 'his', 'thus', 'less', '’m', 'every', 'anyhow', 'although', 'everywhere', 'such', 'even', 'almost', '’d', 'many', 're', 'six', 'before', 'being', 'few', 'thereby', 'yours', 'you', 'quite']