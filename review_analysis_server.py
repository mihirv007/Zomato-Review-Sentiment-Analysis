from flask import Flask, render_template, request
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import joblib
import re
import os
import nltk

# --- SAFE NLTK SETUP FOR SERVERLESS ---
try:
    nltk_data_dir = '/tmp/nltk_data'
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
except Exception as e:
    print(f"NLTK download error: {e}")
# -------------------------------------

# --- ABSOLUTE PATH SETUP FOR VERCEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Load Vectorizer and classifier safely
try:
    vectorizer_path = os.path.join(BASE_DIR, 'review_analysis_tfidVectorizer.pkl')
    classifier_path = os.path.join(BASE_DIR, 'review_analysis_classifier.pkl')
    
    Vectorizer = joblib.load(vectorizer_path)
    Classifier = joblib.load(classifier_path)
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def upload_review():
    return render_template('review_analysis.html')

@app.route('/upload', methods=['POST'])
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

        X_new = Vectorizer.transform([text_analysis])
        prediction = Classifier.predict(X_new)

        predicted = 'Liked' if prediction[0] == 1 else 'not liked'
        return render_template('review_analysis.html', prediction=predicted)
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)