# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:47:24 2024

@author: sgeorgiou
"""


import xml.etree.ElementTree as ET
import os 
from tqdm import tqdm
import re

fileloc = "C:\\Users\\sgeorgiou\\OneDrive - Institution of Engineering and Technology\\Documents\\ChemIndexing\\testitems\\files\\"
#file = "23crq5r3p6hy7.xml"
    
    
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

# def chem_extract_test(xml_file): # this is a test version for single xml files

#     def clean_tags(text):
#         # Define a regular expression pattern to match XML tags
#         pattern = re.compile(r'<[^>]+>')
        
#         # Use sub method to replace all matches of the pattern with an empty string
#         cleaned_text = re.sub(pattern, '', text)
        
#         return cleaned_text
#     ind_dict = {}
    
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
    
#     children = [child for child in root[1][0]]
#     annots = children[-1]
    
    
    
#     chems = []
#     roles = []
#     states = []
#     for annot in annots:
#         if annot.find('{http://data.iet.org/schemas/annotation}references') is not None:
#             ref = annot.find('{http://data.iet.org/schemas/annotation}references')
#             if ref.attrib['schemeLabel'] == "Chemical indexing":
#                 label = clean_tags(ref.attrib['label'])
#                 label = label.split('/')
#                 state = (annot.find('{http://data.iet.org/schemas/annotation}state').attrib['label'])
                
#                 chems.append(label[0])
#                 roles.append(label[1])
#                 states.append(state)
                            
#     ind_dict[xml_file[:-4]] = {'chem':chems, 'role':roles, 'state':states}
    
#     return ind_dict

chem_dict = chem_extract(fileloc)
#%%
chem_counts = 0
non_chem = 0
nonchem = []
nonchem_abs = []
chem = []
chem_abs = []
chem_subs = []
nonchem_subs = []
for item in chem_dict: #loop through items in dictionary
    entry = chem_dict[item]
    if "Added" not in entry['states'] and "AddedSevere" not in entry['states'] and "Accepted" not in entry['states']: #check if only rejected terms appear in the states list
        non_chem += 1
        nonchem.append(item)
        entry['label'] = 'nonchem' #append a new key to the details dictionary for the label; chem or non chem
        nonchem_abs.append(entry['title']+" "+entry['abstract']) #append title and abstract to a list 
        nonchem_subs.append(entry['subject'])
    else:
        chem_counts += 1
        chem.append(item)
        entry['label'] = 'chem'
        chem_abs.append(entry['title']+" "+entry['abstract'])
        chem_subs.append(entry['subject'])

import pandas as pd

df = pd.DataFrame.from_dict(chem_dict,orient="index").reset_index()

    
        
#%%

import clean_text #import clean_text (Calebs code)

#cleaned_chem = []
def clean_abstract_output(chemdict, chemtype): 
    if chemtype != "chem" and chemtype != "nonchem":
        raise TypeError("Chemtype must be 'chem' or 'nonchem' exactly")
        
    cleaned = []
    for item in tqdm(chem_dict):
        if chem_dict[item]['label'] == chemtype:
            clean = clean_text.cleantext(chem_dict[item]['abstract'], "alpha", "lemma")
            
            clean_lower = clean.cleaned.lower()
            clean_lower = re.sub('[.]','',clean_lower)
        
            clean_final = " ".join(clean_lower.split())
            cleaned.append(clean_final)
            
    with open('%s_abstracts.txt'%chemtype, "w") as f:
        for ab in cleaned:
            f.write(ab + ', %s'%chemtype + '\n')


#clean_abstract_output(chem_dict, "chem")
clean_abstract_output(chem_dict, "nonchem")



