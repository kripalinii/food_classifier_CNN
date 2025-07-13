#  Food Image Classifier with CNN

A beginner-friendly **Convolutional Neural Network (CNN)** that recognizes 4 types of food from images: Pizza , Pasta , Salad , and Donuts !


```
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
```

##  Model Architecture

```python
# Simple but effective CNN
Conv2D(32) → MaxPool → BatchNorm →
Conv2D(64) → MaxPool → BatchNorm →
Conv2D(128) → MaxPool → BatchNorm →
Flatten → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(4)
