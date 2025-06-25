# 🍽️ Food Image Classifier with CNN

A fun, beginner-friendly **Convolutional Neural Network (CNN)** that recognizes 4 types of food from images: Pizza 🍕, Pasta 🍝, Salad 🥗, and Donuts 🍩!

Perfect for machine learning portfolios and learning CNN fundamentals with a clean, interactive web interface.

![Food Classifier Demo](https://via.placeholder.com/800x400/667eea/ffffff?text=Food+Image+Classifier+Demo)

## ✨ Features

- 🧠 **Custom CNN Architecture** - Simple but effective 3-layer CNN
- 📊 **Confidence Meter** - Visual confidence scoring with fun messages
- 👨‍🍳 **Chef's Tips** - Random cooking tips for each food type
- 🔗 **Recipe Links** - Direct links to cooking instructions
- 👃 **Easter Eggs** - Fun "smell detector" and hidden features
- 📱 **Web Interface** - Clean, responsive Flask web app
- 🎯 **Resume-Ready** - Well-documented, professional code structure

## 🚀 Quick Start

### 1. Clone and Setup
\`\`\`bash
git clone <your-repo-url>
cd food-classifier
pip install tensorflow numpy matplotlib pillow opencv-python flask scikit-learn
\`\`\`

### 2. Generate Sample Data
\`\`\`bash
python scripts/generate_sample_data.py
\`\`\`

### 3. Train the Model
\`\`\`bash
python model/train_model.py
\`\`\`

### 4. Test Predictions
\`\`\`bash
python model/predict.py
\`\`\`

### 5. Launch Web App
\`\`\`bash
cd webapp
python app.py
\`\`\`

Open http://localhost:5000 in your browser! 🎉

## 📁 Project Structure

\`\`\`
food-classifier/
├── data/                    # Training images (auto-generated)
│   ├── pizza/
│   ├── pasta/
│   ├── salad/
│   └── donuts/
├── model/                   # ML model files
│   ├── train_model.py      # CNN training script
│   ├── predict.py          # Prediction with fun features
│   └── food_classifier.h5  # Saved model (after training)
├── utils/                   # Utility functions
│   └── preprocess.py       # Image preprocessing
├── webapp/                  # Flask web interface
│   ├── app.py             # Flask application
│   ├── templates/         # HTML templates
│   └── static/            # CSS, JS, uploads
├── scripts/               # Helper scripts
│   └── generate_sample_data.py
└── README.md
\`\`\`

## 🧠 Model Architecture

```python
# Simple but effective CNN
Conv2D(32) → MaxPool → BatchNorm →
Conv2D(64) → MaxPool → BatchNorm →
Conv2D(128) → MaxPool → BatchNorm →
Flatten → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(4)
