# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:45:29 2020

@author: Tyler
"""

import pandas as pd 
import numpy as np
import matplotlib
import nltk
from nltk.tokenize import RegexpTokenizer
#the downloader is important for multiple packages
#read on.
nltk.download()
# Load library
from nltk.corpus import stopwords
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
#Set File Path
textmine = pd.read_csv("C:/Users/John/Desktop/CIS Tutorials/Cis3339/covid19_Tweets.csv",sep=',')
#Clean file so the computer can read the data we want read easier
cleanTextMineDF=textmine.drop(['user_name','user_location','user_description','user_created','user_followers','user_friends','user_favourites',
                           'user_verified','date','hashtags','source'],axis=1)

stop_words = stopwords.words('english')
extraSW=('How','What','I','the','may','to','a','in','Why','is','get','Which','why','is','Is','would','If','https','CO','The','Covid','covid','co','coronavirus','Coronavirus',)
##extraSW=('How','What','I','the','may','to','a','in','Why','is','get','Which','why','is','Is','would','If')
for i in range(len(extraSW)):
 stop_words.append(extraSW[i])
 
stop_words[:0]

texttotokens=[]

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

for i in range(cleanTextMineDF.shape[0]):
 texttotoken=cleanTextMineDF.text[i]
 texttotokens.append(tokenizer.tokenize(texttotoken))
 
flat_list = []
for sublist in texttotokens:
    for item in sublist:
        flat_list.append(item)
        
frequency_dist = nltk.FreqDist([word for word in tokenizer.tokenize(str(flat_list)) if word not in stop_words]) 
top50n=sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]
numberoftimes=[]
for i in range(50):
 numberoftimes.append(frequency_dist[top50n[i]])       

wordcloud = WordCloud(stopwords=stop_words).generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud, interpolation='bilinear') 


subsetted=pd.DataFrame()        
def subsetter(datasetname,idc,value):   
 newdatasetname=datasetname[datasetname[idc]==value]
 return newdatasetname      
subsetted=subsetter(cleanTextMineDF,'is_retweet',0) 
subsetted=subsetter(cleanTextMineDF,'is_retweet',1)

texttotokens=[]
for i in range(subsetted.shape[0]):
 texttotoken=subsetted.text.iloc[i]
 texttotokens.append(tokenizer.tokenize(texttotoken)) 

 
wordcloud0=WordCloud(stopwords=stop_words).generate_from_frequencies(frequency_dist)
wordcloud1=WordCloud(stopwords=stop_words).generate_from_frequencies(frequency_dist)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.imshow(wordcloud0, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
plt.show()