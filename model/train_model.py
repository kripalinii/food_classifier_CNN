"""
CNN Model Training Script for Food Image Classifier
Trains a simple but effective CNN to recognize 4 types of food
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.append('..')
from utils.preprocess import FoodImagePreprocessor

def create_cnn_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Create a simple but effective CNN model for food classification
    
    Args:
        input_shape (tuple): Shape of input images
        num_classes (int): Number of food classes to predict
        
    Returns:
        tensorflow.keras.Model: Compiled CNN model
    """
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')  # 4 food classes
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot training accuracy and loss curves
    
    Args:
        history: Keras training history object
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax1.set_title('Model Accuracy 📈')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2.set_title('Model Loss 📉')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Training plots saved as 'training_history.png'")

def train_food_classifier():
    """
    Main training function - loads data, creates model, and trains it
    """
    
    print("🍕 Starting Food Image Classifier Training! 🍝")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = FoodImagePreprocessor(img_size=(128, 128))
    
    # Load dataset
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print("❌ Data directory not found!")
        print("💡 Run 'python scripts/generate_sample_data.py' first to create sample data")
        return
    
    images, labels = preprocessor.load_dataset(data_dir)
    
    if len(images) == 0:
        print("❌ No images found in dataset!")
        return
    
    # Shuffle the data
    images, labels = shuffle(images, labels, random_state=42)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(X_train)} images")
    print(f"   Validation: {len(X_val)} images")
    
    # Create and display model
    model = create_cnn_model()
    print("\n🧠 Model Architecture:")
    model.summary()
    
    # Set up callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train the model
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("\n📊 Final Evaluation:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"   Training Accuracy: {train_acc:.3f}")
    print(f"   Validation Accuracy: {val_acc:.3f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'food_classifier.h5'))
    
    print(f"\n✅ Model saved to '{model_dir}/food_classifier.h5'")
    print("🎉 Training completed successfully!")
    
    return model, history

if __name__ == "__main__":
    model, history = train_food_classifier()

# Important Variables:
# - input_shape: Dimensions of input images (128x128x3)
# - num_classes: Number of food categories (4)
# - batch_size: Number of images processed together (32)
# - epochs: Maximum training iterations (20)

# Major Functions:
# - create_cnn_model(): Builds the CNN architecture
# - plot_training_history(): Visualizes training progress
# - train_food_classifier(): Main training pipeline
