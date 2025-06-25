"""
Image Preprocessing Utilities for Food Classifier
Handles loading, resizing, and augmenting food images
"""

import os
import numpy as np
import cv2
from PIL import Image
import random

class FoodImagePreprocessor:
    """
    Handles all image preprocessing tasks for the food classifier
    """
    
    def __init__(self, img_size=(128, 128)):
        """
        Initialize preprocessor with target image size
        
        Args:
            img_size (tuple): Target size for all images (width, height)
        """
        self.img_size = img_size
        self.class_names = ['pizza', 'pasta', 'salad', 'donuts']
        
    def load_and_preprocess_image(self, img_path):
        """
        Load a single image and preprocess it for the model
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Load image using PIL (handles various formats)
            img = Image.open(img_path)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize(self.img_size)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            return None
    
    def load_dataset(self, data_dir):
        """
        Load all images from the dataset directory
        
        Args:
            data_dir (str): Path to the data directory containing class folders
            
        Returns:
            tuple: (images, labels) as numpy arrays
        """
        images = []
        labels = []
        
        print("📂 Loading dataset...")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  Warning: {class_dir} not found, skipping...")
                continue
            
            # Get all image files in the class directory
            img_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"   Loading {len(img_files)} {class_name} images...")
            
            for img_file in img_files:
                img_path = os.path.join(class_dir, img_file)
                img_array = self.load_and_preprocess_image(img_path)
                
                if img_array is not None:
                    images.append(img_array)
                    labels.append(class_idx)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"✅ Dataset loaded: {len(images)} images, {len(set(labels))} classes")
        
        return images, labels
    
    def augment_image(self, img_array):
        """
        Apply simple data augmentation to an image
        
        Args:
            img_array (numpy.ndarray): Input image array
            
        Returns:
            numpy.ndarray: Augmented image array
        """
        # Convert to PIL for easier augmentation
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(255, 255, 255))
        
        # Convert back to normalized array
        return np.array(img) / 255.0

# Important Variables:
# - img_size: Target dimensions for all processed images
# - class_names: List of food categories the model will recognize

# Major Functions:
# - load_and_preprocess_image(): Loads and normalizes a single image
# - load_dataset(): Loads entire dataset from directory structure
# - augment_image(): Applies random transformations for data augmentation
