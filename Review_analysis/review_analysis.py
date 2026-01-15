import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score
import joblib

#75% accuracy

#loding the dataset
dataset= pd.read_csv("/Users/mihirverma/Review_analysis/Review_analysis.csv")


corpus=[]

dataset = dataset.dropna(subset=['Review', 'Liked'])
dataset.reset_index(drop=True, inplace=True)

#data cleaning

for i in range(len(dataset)):
    review_analysis=re.sub('[^a-zA-Z]' ,' ' ,dataset['Review'][i])
    
    review_analysis=review_analysis.lower()
    
    review_analysis=review_analysis.split()

    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    
    review_analysis=[ps.stem(word) for word in review_analysis if word not in all_stopwords and len(word)>2]
    
    review_analysis=' '.join(review_analysis)
    
    corpus.append(review_analysis)
   

#train and testing datasets
cv=TfidfVectorizer(max_features=3000,ngram_range=(1,2))
X=cv.fit_transform(corpus).toarray()
y=dataset['Liked'].values

#splitting validation and training datasets

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)

Classifier=MultinomialNB(alpha=0.5)

Classifier.fit(X_train,y_train)

y_pred=Classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)
acs=accuracy_score(y_test,y_pred)
print(acs)

#save the vectorizer and classifier

joblib.dump(Classifier,'review_analysis_classifier.pkl')
joblib.dump(cv,'review_analysis_tfidVectorizer.pkl')