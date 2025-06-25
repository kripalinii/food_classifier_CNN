"""
Flask Web App for Food Image Classifier
Simple, clean interface for uploading images and getting predictions
"""

from flask import Flask, render_template, request, jsonify, url_for
import os
import sys
from werkzeug.utils import secure_filename
import uuid

# Add parent directory to path for imports
sys.path.append('..')
from model.predict import FoodPredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'food-classifier-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
predictor = FoodPredictor(model_path='../model/food_classifier.h5')

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save uploaded file
        file.save(filepath)
        
        # Make prediction
        result = predictor.predict_food(filepath)
        
        if 'error' not in result:
            # Add image URL for display
            result['image_url'] = url_for('static', filename=f'uploads/{filename}')
            
            # Clean up - remove uploaded file after prediction
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/about')
def about():
    """About page with project info"""
    return render_template('about.html')

if __name__ == '__main__':
    if predictor.model is None:
        print("❌ Model not found! Please train the model first:")
        print("   python model/train_model.py")
    else:
        print("🚀 Starting Food Classifier Web App...")
        print("📱 Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)

# Important Variables:
# - ALLOWED_EXTENSIONS: Supported image file types
# - UPLOAD_FOLDER: Directory for temporary file storage
# - predictor: Global FoodPredictor instance

# Major Functions:
# - index(): Serves the main upload page
# - predict(): Handles file upload and prediction
# - about(): Serves project information page
