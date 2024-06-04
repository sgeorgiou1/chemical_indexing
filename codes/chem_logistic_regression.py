# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:00:47 2024

@author: sgeorgiou
"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import random
import pandas as pd

data = pd.read_csv("./abstracts/labelled_abstracts.csv")
# Shuffle all rows in the DataFrame
shuffled_df = data.sample(frac=1, random_state=42)

# Reset the index if needed
shuffled_df.reset_index(drop=True, inplace=True)

X = shuffled_df['Text']
y = shuffled_df['Label']


#%%
from sklearn.model_selection import cross_val_score

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=200000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train logistic regression model
logreg_model = LogisticRegression(C=10.0)
logreg_model.fit(X_train, y_train)

# Make predictions
y_pred = logreg_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
scores = cross_val_score(logreg_model, X_tfidf,y, scoring = "accuracy", cv=5)
print (scores)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
import pickle

# Save the vectorizer to disk
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('logreg_model.pkl', 'wb') as file:
    pickle.dump(logreg_model, file)


#%%
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression

# # Define the parameter grid
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Example values for the regularization parameter C
# }
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
# X_tfidf = vectorizer.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# # Create the model
# logreg_model = LogisticRegression(max_iter=1000)

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(estimator=logreg_model, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_params = grid_search.best_params_
# print("Best hyperparameters:", best_params)

# # Get the best model
# best_model = grid_search.best_estimator_

# # Evaluate the best model
# accuracy = best_model.score(X_test, y_test)
# print("Accuracy:", accuracy)