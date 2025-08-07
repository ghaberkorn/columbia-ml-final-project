import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import spacy
from sklearn.model_selection import RandomizedSearchCV

trueDF = pd.read_csv('/Users/ighacibeh/.cache/kagglehub/datasets/clmentbisaillon/fake-and-real-news-dataset/versions/1/True.csv')
falseDF = pd.read_csv('/Users/ighacibeh/.cache/kagglehub/datasets/clmentbisaillon/fake-and-real-news-dataset/versions/1/True.csv')

# Drop NAN values, and 'date' column
cleaned_trueDF = trueDF.dropna()
cleaned_trueDF = cleaned_trueDF.drop(columns = ['date'])

cleaned_falseDF = falseDF.dropna()
cleaned_falseDF = cleaned_falseDF.drop(columns = ['date'])

# Make values lower case
cleaned_trueDF['title'] = cleaned_trueDF['title'].str.lower()
cleaned_trueDF['text'] = cleaned_trueDF['text'].str.lower()
cleaned_trueDF['subject'] = cleaned_trueDF['subject'].str.lower()

cleaned_falseDF['title'] = cleaned_falseDF['title'].str.lower()
cleaned_falseDF['text'] = cleaned_falseDF['text'].str.lower()
cleaned_falseDF['subject'] = cleaned_falseDF['subject'].str.lower()

# Filter out filler words, 'the', 'a', 'an', 'and', 'or', 'but'
filler_words = ['the', 'a', 'an']

def remove_filler(text):
    return ' '.join([word for word in text.split() if word.lower() not in filler_words])

cleaned_trueDF['text'] = cleaned_trueDF['text'].apply(remove_filler)
cleaned_falseDF['text'] = cleaned_falseDF['text'].apply(remove_filler)


# split into features and labels    
X_true = cleaned_trueDF
y_true = np.ones(len(X_true), dtype=bool)    # creates an arry of true

X_false = cleaned_falseDF
y_false = np.zeros(len(X_false), dtype=bool) # creates an array of false

# concatenate X and y
X = pd.concat([X_true, X_false], ignore_index=True)
y = np.concatenate([y_true, y_false])

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_labels = ["Fake", "Real"]

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_train_tf = tfidf.fit_transform(X_train['text'])  # Extract 'text' column for vectorization

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train_tf, y_train)

from sklearn.metrics import accuracy_score

X_test_tf = tfidf.transform(X_test['text'])  # Extract 'text' column for vectorization

y_pred = model.predict(X_test_tf)
print ('Accuracy Score - ', accuracy_score(y_test, y_pred))

from sklearn.dummy import DummyClassifier

clf = DummyClassifier(strategy='most_frequent')
clf.fit(X_train, y_train)
y_pred_baseline = clf.predict(X_test)
print ('Accuracy Score - ', accuracy_score(y_test, y_pred_baseline))

test_text = [""]

predicted_sentiment = model.predict(tfidf.transform(test_text))
print(predicted_sentiment)

from sklearn.metrics import recall_score, precision_score

print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))

param_grid = {
    'penalty': ['l1', 'l2'],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'max_iter': [1, 3000]
}

lsvc = LinearSVC(random_state=42)

random_search = RandomizedSearchCV(
    estimator=lsvc,
    param_distributions=param_grid,
    n_iter=1000,  # Number of parameter settings to sample
    cv=5,        # Number of cross-validation folds
    scoring='accuracy',
    random_state=42, # For reproducibility of the random sampling
)
random_search.fit(X_train_tf, y_train)

print("Best parameters:", random_search.best_params_)
model_improved = random_search.best_estimator_

model_improved.fit(X_train_tf, y_train)

y_pred = model_improved.predict(X_test_tf)
print ('Accuracy Score - ', accuracy_score(y_test, y_pred))

print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))

# Using the original dataset accuracy, precision, and recall somehow all went down when using these hyperparameters
# Using a new dataset, the model performed worse than the dummy classifier. The hyperparameters helped, but it still performed horribly
# There is a good chance this model is overfitting