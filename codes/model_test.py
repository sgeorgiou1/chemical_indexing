# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:46:06 2024

@author: sgeorgiou
"""
import xml.etree.ElementTree as ET
import os 
from tqdm import tqdm
import re
from chem_functions import chem_extract
from chem_functions import clean_abstract_output
import pandas as pd
#%%
#fileloc = "C:\\Users\\sgeorgiou\\OneDrive - Institution of Engineering and Technology\\Documents\\ChemIndexing\\testitems\\files\\"
fileloc = "C:\\Users\\sgeorgiou\\OneDrive - Institution of Engineering and Technology\\Documents\\ChemIndexing\\NonchemPapers\\files\\"
chem_dict = chem_extract(fileloc)

chem_counts = 0
non_chem = 0
nonchem = []
nonchem_abs = []
chem = []
chem_abs = []
for item in chem_dict: #loop through items in dictionary
    entry = chem_dict[item]
    if "Added" not in entry['states'] and "AddedSevere" not in entry['states'] and "Accepted" not in entry['states']: #check if only rejected terms appear in the states list
        non_chem += 1
        nonchem.append(item)
        chem_dict[item]['label'] = 'nonchem' #append a new key to the details dictionary for the label; chem or non chem
        nonchem_abs.append(entry['title']+" "+entry['abstract']) #append title and abstract to a list 
    else:
        chem_counts += 1
        chem.append(item)
        chem_dict[item]['label'] = 'chem'
        chem_abs.append(entry['title']+" "+entry['abstract'])

clean_chem = clean_abstract_output(chem_dict, "chem", "test")
clean_nonchem = clean_abstract_output(chem_dict, "nonchem", "test")

chem_df = pd.DataFrame.from_dict(chem_dict, orient='index').reset_index()

#%% LOAD PICKLE FILES

import pickle
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open('./models/vectorizer.pkl', 'rb') as vec:
    vectorizer = pickle.load(vec)

with open("./models/logreg_model.pkl", 'rb') as mod:
    logreg = pickle.load(mod)
    
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = logreg.coef_[0]

# Map coefficients to feature names
feature_importance = list(zip(feature_names, coefficients))

# Sort feature importance by coefficient magnitude
sorted_feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

# Print top N features and their coefficients
top_n = 10
for feature, coef in sorted_feature_importance[:top_n]:
    print(f"Feature: {feature}, Coefficient: {coef}")

# Create a dictionary of feature names and their corresponding coefficients
wordcloud_chemdict = {feature: coef for feature, coef in sorted_feature_importance}
wordcloud_nonchemdict = {feature: coef*-1 for feature, coef in sorted_feature_importance}
# Generate word cloud
wordcloud_nonchem = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_chemdict)
wordcloud_chem = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_nonchemdict)
# Display the word cloud
plt.figure(figsize=(16, 8), dpi = 400)

plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot 1
plt.imshow(wordcloud_chem, interpolation='bilinear')
plt.title('Chemical Important Features')
plt.axis('off')

plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot 2
plt.imshow(wordcloud_nonchem, interpolation='bilinear')
plt.title('Nonchemical Important Features')
plt.axis('off')

plt.show()

#%%
#test_tfidf_chem = vectorizer.transform(clean_chem['cleaned'])
#predicted_chem = logreg.predict(test_tfidf_chem)

test_tfidf_nonchem = vectorizer.transform(clean_nonchem['cleaned'])
predicted_nonchem = logreg.predict(test_tfidf_nonchem)
#%%

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


#pred_chem = [l.strip() for l in predicted_chem]
pred_nonchem = [l.strip() for l in predicted_nonchem]

#true_chem = ["chem" for i in range(len(pred_chem))]
true_nonchem = ["nonchem" for i in range(len(pred_nonchem))]

id_list = clean_chem['ids'].to_list() + clean_nonchem['ids'].to_list()

#output_df = pd.DataFrame(list(zip(id_list,pred_chem+pred_nonchem,true_chem+true_nonchem)),columns = ["IDs","Predicted","True"])
output_df = pd.DataFrame(list(zip(id_list,pred_nonchem,true_nonchem)),columns = ["IDs","Predicted","True"])
shuffled_df = shuffle(output_df)

matrix = confusion_matrix(shuffled_df["Predicted"],shuffled_df["True"])

disp = ConfusionMatrixDisplay(matrix,display_labels=['Chem','Nonchem'])
disp.plot()
plt.show()

chem_fscore = f1_score(shuffled_df["Predicted"],shuffled_df["True"],pos_label="chem")
nonchem_fscore = f1_score(shuffled_df["Predicted"],shuffled_df["True"],pos_label="nonchem")
print ("Fscore if chem is positive: ",chem_fscore)
print ("Fscore if nonchem is positive: ",nonchem_fscore)
