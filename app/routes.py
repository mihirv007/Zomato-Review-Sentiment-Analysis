from flask import Blueprint, render_template, request, current_app
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def upload_review():
    return render_template('review_analysis.html')

@main_bp.route('/upload', methods=['POST'])
def predict():
    try:
        text = request.form['review']
        text_analysis = re.sub('[^a-zA-Z]', ' ', text)
        text_analysis = text_analysis.lower()
        text_analysis = text_analysis.split()

        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        if 'not' in all_stopwords:
            all_stopwords.remove('not')

        text_analysis = [ps.stem(word) for word in text_analysis if word not in all_stopwords and len(word) > 2]
        text_analysis = ' '.join(text_analysis)

        # Grab models from app config
        vectorizer = current_app.config['VECTORIZER']
        classifier = current_app.config['CLASSIFIER']

        X_new = vectorizer.transform([text_analysis])
        prediction = classifier.predict(X_new)

        predicted = 'Liked' if prediction[0] == 1 else 'not liked'
        return render_template('review_analysis.html', prediction=predicted)
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}", 500