# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:10:37 2024

@author: sgeorgiou
"""
import re
import xml.etree.ElementTree as ET
import os 
from tqdm import tqdm
import pandas as pd
import clean_text #import clean_text (Calebs code)
import pickle

def chem_extract(xml_location): # extract chemical indexing from inspec2 xml messages
    def clean_tags(text):
        # Define a regular expression pattern to match XML tags
        pattern = re.compile(r'<[^>]+>') #this cleans up xml tags eg <sub></sub>
        
        # Use sub method to replace all matches of the pattern with an empty string
        cleaned_text = re.sub(pattern, '', text)
        
        return cleaned_text
    
    ind_dict = {} #create a blank dictionary to store indexing info
    for file in tqdm(os.listdir(xml_location)):
        tree = ET.parse(xml_location+file) #parse the xml
        root = tree.getroot()
        
        children = [child for child in root[1][0]] #split parsed xml into main child nodes root[1][0] is the data section
        
        title = root[1][0].find('{http://data.iet.org/schemas/inspec/content}title').text #extract title text
        title = clean_tags(title) # clean xml tags from title text
        
        abstract = root[1][0].find('{http://data.iet.org/schemas/inspec/content}abstract').text #same for abstract
        abstract = clean_tags(abstract)
        
        subject = root[1][0].find('{http://data.iet.org/schemas/inspec/content}specialisationType').get("label")
        
        annots = children[-1] #annotations node is final node in the child nodes
        
        
        
        chems = []
        roles = []
        states = []
        for annot in annots: #loop through all annotations and search for chemical indexing label
            if annot.find('{http://data.iet.org/schemas/annotation}references') is not None:
                ref = annot.find('{http://data.iet.org/schemas/annotation}references')
                if ref.attrib['schemeLabel'].lower() == "chemical indexing": #chem indexing label is in references/scheme label
                    label = clean_tags(ref.attrib['label']) # this returns the chemical label with its role split by a "/" eg SiO2/bin
                    label = label.split('/')
                    state = (annot.find('{http://data.iet.org/schemas/annotation}state').attrib['label']) # returns annotation state (rejected/accepted etc.)
                    
                    chems.append(label[0])
                    roles.append(label[1])
                    states.append(state)
                                
        ind_dict[file[:-4]] = {'title':title, 'abstract':abstract, 'subject': subject, 'chems':chems, 'roles':roles, 'states':states} #create dictionary entry for all info, item id is the key, value is a dictionary of details title,abs (strs) and chems,roles,states (lists)
    
    return ind_dict






def clean_abstract_output(chem_dict, chemtype, write="y", file_out_name=None): 
    if chemtype != "chem" and chemtype != "nonchem":
        raise TypeError("Chemtype must be 'chem' or 'nonchem' exactly")
    ids = []
    cleaned = []
    for item in tqdm(chem_dict):
        if chem_dict[item]['label'] == chemtype:
            clean = clean_text.cleantext(chem_dict[item]['abstract'], "alpha", "lemma")
            
            clean_lower = clean.cleaned.lower()
            clean_lower = re.sub('[.]','',clean_lower)
        
            clean_final = " ".join(clean_lower.split())
            cleaned.append(clean_final)
            
            ids.append(item)
    if write.lower == "y":        
        with open('%s_abstracts_'%chemtype +file_out_name +'.txt', "w") as f:
            for ab in cleaned:
                f.write(ab + ', %s'%chemtype + '\n')
    return pd.DataFrame(list(zip(ids,cleaned)),columns = ["ids","cleaned"])


def chem_predict(abstract):
    with open('./models/vectorizer.pkl', 'rb') as vec:
        vectorizer = pickle.load(vec)

    with open("./models/logreg_model.pkl", 'rb') as mod:
        logreg = pickle.load(mod)
        
    cleaned = clean_text.cleantext(abstract, 'alpha', 'lemma')
    
    tfidf = vectorizer.transform([cleaned.cleaned])
    prediction = logreg.predict(tfidf)
    
    return prediction[0].strip()