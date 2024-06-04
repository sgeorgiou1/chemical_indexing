# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:02:56 2021

@author: csevern
"""
import subprocess
import sys
import re
import nltk

dler = nltk.downloader.Downloader()
dler._update_index()
try:
    from nltk.corpus import stopwords  
except:
    dler.download("stopwords")
    from nltk.corpus import stopwords
try:
    from nltk import pos_tag
except:
    dler.download('averaged_perceptron_tagger') 

    
try:
    from nltk.stem import WordNetLemmatizer
except:
    dler.download('wordnet')
    from nltk.stem import WordNetLemmatizer           

try:
    from stemming.porter2 import stem
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "stemming"])         
    from stemming.porter2 import stem 
stop_words = set(stopwords.words('english'))  
lr = WordNetLemmatizer()

notice = """
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Clean Text: type clean_text.info() to see what you can do
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""  
print(notice)  

def info():
    notice = """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    This is the clean_text program written by Caleb
    Please make sure you have nltk installed
    Use this programs functions listed below
    lemma - lemmatizes your words
    clean_alpha_num - cleans everything that is not alphanumeric or . -
    clean_alpha - cleans everything that is not alpha or . -
    clean_accent - cleans out all accents
    tagger - returns POS tags in list
    stemmed - returns stemmed words in list
    
    Use the main class cleantext(text,clean,lem_port)
    Where:
    text - the text you want to be cleaned/stemmed
    clean - "alpha_num","alpha","accent","None"
    lem_port -"lemma","port","None"
    
    Then you can see any attribute of your text
    .cleaned - cleaned text as string
    .lem - lemmatized if selected
    .port - port stemmed if selected
    .tags - part of speech tagging
    .stopw - without stopwords
    .words - words as list
    .sents - as a list
    
    the functions are applied in the order below
    cleaned->stopw->tags->lem/stem->words->sents

    type clean_text.info() to get this info again
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """  
    print(notice)


def lemma(text):
    if type(text) != list:
        text = nltk.word_tokenize(text)
    for i,t in enumerate(text):
        lword = lr.lemmatize(text[i])
        text[i] = lword
    text= ' '.join(text)
    return text;

def clean_alpha_num(text):
    if type(text) == list:
        text = ' '.join(text)
    textc = re.sub("[^0-9a-zA-Z-. ]+",'', text)
    textc = re.sub("-"," ",textc)
    return textc;

def clean_alpha(text):
    if type(text) == list:
        text = ' '.join(text)
    textc = re.sub("[^a-zA-Z-. ]+",'', text)
    return textc;

def clean_accent(text):
    if type(text) == list:
        text = ' '.join(text)
    textc = re.sub("[^a-zA-Zα-ωΑ-Ω0-9-?!. ]+",'', text)
    return textc;    

def tagger(text):
    if type(text) != list:
        text = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(text)
    tags = []
    for i,v in tagged:
        tags.append(v)
    return tags;

def stemmed(text):
    if type(text) != list:
        text = nltk.word_tokenize(text)
    
    stems = [stem(x) for x in text]
    text= ' '.join(stems)
    return text;
def stops(text):
    if type(text) != list:
        text = nltk.word_tokenize(text)    
    stopw= [x for x in text if x.lower() not in stop_words]
    stopws = ' '.join(stopw)
    return stopws

class cleantext():

    def __init__(self,text,clean,lem_port):
        if clean == "":
            clean = "accent"
        if lem_port =="":
            lem_port = "lemma"        
        if clean == "alpha_num":
            self.cleaned = clean_alpha_num(text)
        if clean == "alpha":
            self.cleaned = clean_alpha(text)
        if clean == "accent":
            self.cleaned = clean_accent(text)
        if clean == "None":
            self.cleaned = text
        self.cleaned = stops(self.cleaned)
        if lem_port == "lemma":
        
            self.lem = lemma(self.cleaned) 
            self.tags = tagger(self.cleaned)
            self.stopw = self.cleaned
            self.words = nltk.word_tokenize(self.lem)
            self.sents = nltk.sent_tokenize(self.lem)
        if lem_port == "port":
            self.port= stemmed(self.cleaned) 
            self.tags = tagger(self.cleaned)
            self.stopw = self.cleaned
            self.words = nltk.word_tokenize(self.port)
            self.sents = nltk.sent_tokenize(self.port)

        if lem_port == "None":
            self.tags = tagger(self.cleaned)
            self.stopw = self.cleaned
            self.words = nltk.word_tokenize(self.cleaned)
            self.sents = nltk.sent_tokenize(self.cleaned)                        


                

