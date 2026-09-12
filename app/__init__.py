import os
from flask import Flask
import joblib
import nltk

def create_app():
    app = Flask(__name__)
    
    # Absolute path for current app directory
    basedir = os.path.abspath(os.path.dirname(__file__))
    
    # Configure template and static folders explicitly
    app.template_folder = os.path.join(basedir, 'templates')
    app.static_folder = os.path.join(basedir, 'static')
    
    # Safe NLTK setup
    try:
        nltk_data_dir = '/tmp/nltk_data'
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        print(f"NLTK error: {e}")

    # Load models from app/models/
    try:
        vectorizer_path = os.path.join(basedir, 'models', 'review_analysis_tfidVectorizer.pkl')
        classifier_path = os.path.join(basedir, 'models', 'review_analysis_classifier.pkl')
        
        app.config['VECTORIZER'] = joblib.load(vectorizer_path)
        app.config['CLASSIFIER'] = joblib.load(classifier_path)
    except Exception as e:
        print(f"Model loading error: {e}")

    # Import routes/views here to avoid circular imports
    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app