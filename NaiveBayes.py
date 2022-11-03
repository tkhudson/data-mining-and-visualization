import pandas as pd 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#conda install -c conda-forge spacy
#pip install scikit-learn
#pip install spacy
#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
#python -m spacy download en_core_web_sm
import spacy
spacy.load('en_core_web_sm')
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score


lemmatizer = spacy.lang.en.English()
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stop_words = stopwords.words('english')
####################do the same target 0 and 1




QuoraQ = pd.read_csv("C:/Users/tyler/OneDrive/Documents/school2k20/Fall2020/CIS3339/FINAL/traintextMining.csv", sep= ',') 
# create a spaCy tokenizer
# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    [token.lemma_ for token in tokens]
    return tokenizer.tokenize(str(tokens))
sw=my_tokenizer(str(stop_words))

#this is an interesting feature where the object cv is modified by the fit operator.
#we can use the dir to get all the interesting attributes of an object.
#we will then use 

cv = CountVectorizer(lowercase=True, stop_words=sw,binary=True,tokenizer=my_tokenizer)
X_train_cv = cv.fit_transform(QuoraQ.question_text)

#interesting attributes 
#dir(X_train_cv)
Names=cv.get_feature_names()

dict1=cv.vocabulary_

#Just for demo
#X_train_cv[0,0:100].toarray()

y_train=QuoraQ.target

########################################################

naive_bayes = BernoulliNB()
naive_bayes.fit(X_train_cv, y_train)
trainedpred=naive_bayes.predict(X_train_cv)
y_train==trainedpred



predictions = naive_bayes.predict(X_train_cv)
#How many insencere questions are there?
sum(y_train==1)
#this subsets y_train to cases where predictions on the corresponding row is equal to 1.
y_train[predictions==1]
#this subsets to cases where given predictions==1 the target variable y to be ==1
y_train[predictions==1]==1
###How many such cases are there?
sum(y_train[predictions==1]==1)
#out of how many?
sum(y_train[predictions==1]==1)/sum(y_train==1)

X_test = pd.read_csv("C:/Users/tyler/OneDrive/Documents/SCHOOL/school2k20/Fall2020/CIS3339/FINAL/WordCloudTestMineResources/testtextMining.csv", sep= ',').iloc[:,1] 
cvNew=CountVectorizer(vocabulary=cv.vocabulary_ ,lowercase=True, stop_words=sw,binary=True,tokenizer=my_tokenizer)
X_test_cv = cvNew.fit_transform(X_test)


#smaller version for class using 200,000 training 40,000 test for speed and testing
playset=QuoraQ.sample(int(QuoraQ.shape[0]*.2))

pX_train, pX_test, py_train, py_test = train_test_split(playset['question_text'], playset['target'], train_size=int(playset.shape[0]*.7),random_state=1)

#################################################################

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',binary=True,lowercase=True,
strip_accents='unicode',token_pattern=r'[A-Za-z]+',tokenizer=my_tokenizer)

X_train_cv = cv.fit_transform(pX_train)
y_train=py_train
########################################################
from sklearn.naive_bayes import BernoulliNB
naive_bayes = BernoulliNB()
#.fit instead of .fit_transform is necc to ignore the 0 prob
naive_bayes.fit(X_train_cv, y_train)

X_test_cv = cv.transform(pX_test)
predictions = naive_bayes.predict(X_test_cv)

###############################################################################

print('Accuracy score: ', accuracy_score(py_test, predictions))
sum(py_test==predictions)/len(predictions)

print('Precision score: ', precision_score(py_test, predictions))
sum(py_test[predictions==1]==1)/len(py_test[predictions==1])

print('Recall score: ', recall_score(py_test, predictions))
sum(predictions[py_test==1]==1)/len(predictions[py_test==1]==1)

#Specificity
sum(predictions[py_test==0]==0)/len(predictions[py_test==0]==0)


Accuracy 
Precision 
Recall (Sensitivity)
Specificity

Accuracy
py_test==predictions

py_test[predictions==1]==1



