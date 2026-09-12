from flask import Flask,render_template,request
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import joblib
from sklearn.feature_extraction.text import TfidfTransformer
import re
import numpy as np
import os


#load Vectorizer and classifer

Vectorizer=joblib.load('review_analysis_tfidVectorizer.pkl')
Classifier=joblib.load('review_analysis_classifier.pkl')

corpus=[]

app=Flask(__name__)
@app.route('/')
def upload_review():
    return render_template('review_analysis.html')

@app.route('/upload',methods=['POST'])
def predict():
    text=request.form['review']
    text_analysis=re.sub('[^a-zA-Z]',' ',text)
    text_analysis=text_analysis.lower()
    text_analysis=text_analysis.split()

    #covert the word to its root form

    ps=PorterStemmer()

    all_stopwords=stopwords.words('english')

    all_stopwords.remove('not')

    text_analysis=[ps.stem(word) for word in text_analysis if word not in  all_stopwords and len(word)>2]

    text_analysis=' '.join(text_analysis)


    print(text_analysis)

    X_new=Vectorizer.transform([text_analysis])

    prediction=Classifier.predict(X_new)

    if prediction[0]==1:
        predicted='Liked'
    else :
        predicted='not liked'

    return render_template('review_analysis.html',prediction=predicted)



if __name__=='__main__':
    app.run(debug=True)






