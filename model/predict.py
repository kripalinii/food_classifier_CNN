"""
Food Image Prediction Script with Fun Features
Loads trained model and predicts food classes with confidence and chef tips!
"""

import os
import numpy as np
import random
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append('..')
from utils.preprocess import FoodImagePreprocessor

class FoodPredictor:
    """
    Fun food prediction class with chef tips and confidence meter
    """
    
    def __init__(self, model_path='model/food_classifier.h5'):
        """
        Initialize the predictor with trained model
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.class_names = ['pizza', 'pasta', 'salad', 'donuts']
        self.class_emojis = ['🍕', '🍝', '🥗', '🍩']
        self.preprocessor = FoodImagePreprocessor()
        
        # Load the trained model
        try:
            self.model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
        
        # Chef's tips for each food class
        self.chef_tips = {
            'pizza': [
                "🍕 Pro tip: Let your pizza dough rest for at least 24 hours for better flavor!",
                "🍕 Fun fact: The first pizza was actually white - no tomatoes!",
                "🍕 Chef's secret: A pinch of sugar in tomato sauce balances the acidity.",
                "🍕 Did you know? Pizza Margherita represents the Italian flag colors!"
            ],
            'pasta': [
                "🍝 Pro tip: Save some pasta water - it's liquid gold for your sauce!",
                "🍝 Fun fact: There are over 300 pasta shapes in Italy!",
                "🍝 Chef's secret: Add pasta to sauce, not sauce to pasta!",
                "🍝 Did you know? Al dente means 'to the tooth' in Italian!"
            ],
            'salad': [
                "🥗 Pro tip: Dress your salad just before serving to keep it crisp!",
                "🥗 Fun fact: Caesar salad was invented in Mexico, not Italy!",
                "🥗 Chef's secret: Massage kale with salt to make it tender.",
                "🥗 Did you know? Iceberg lettuce is 96% water!"
            ],
            'donuts': [
                "🍩 Pro tip: Fry donuts at exactly 375°F for perfect texture!",
                "🍩 Fun fact: The donut hole was invented to cook more evenly!",
                "🍩 Chef's secret: Cake donuts vs yeast donuts - totally different techniques!",
                "🍩 Did you know? National Donut Day is the first Friday of June!"
            ]
        }
        
        # Recipe URLs (hardcoded as requested)
        self.recipe_urls = {
            'pizza': "https://www.allrecipes.com/recipe/213742/cheesy-pizza/",
            'pasta': "https://www.allrecipes.com/recipe/11973/spaghetti-aglio-e-olio/",
            'salad': "https://www.allrecipes.com/recipe/14276/caesar-salad-supreme/",
            'donuts': "https://www.allrecipes.com/recipe/21065/old-fashioned-sour-cream-doughnuts/"
        }
    
    def get_confidence_message(self, confidence, predicted_class):
        """
        Generate fun confidence messages based on prediction certainty
        
        Args:
            confidence (float): Prediction confidence (0-1)
            predicted_class (str): Predicted food class
            
        Returns:
            str: Fun confidence message
        """
        emoji = self.class_emojis[self.class_names.index(predicted_class)]
        
        if confidence >= 0.9:
            return f"I'm {confidence:.1%} sure this is {predicted_class}! {emoji}✅ Crystal clear!"
        elif confidence >= 0.7:
            return f"I'm {confidence:.1%} confident this is {predicted_class}. {emoji}👍 Pretty sure!"
        elif confidence >= 0.5:
            return f"I think this might be {predicted_class} ({confidence:.1%} confidence). {emoji}🤔 Somewhat sure..."
        else:
            return f"Umm... is that {predicted_class} or a saucy lens cap? 🤔 Only {confidence:.1%} confident!"
    
    def get_chef_tip(self, predicted_class):
        """
        Get a random chef tip for the predicted food class
        
        Args:
            predicted_class (str): Predicted food class
            
        Returns:
            str: Random chef tip
        """
        return random.choice(self.chef_tips[predicted_class])
    
    def predict_food(self, image_path):
        """
        Predict food class from image with fun features
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results with fun features
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Preprocess the image
        img_array = self.preprocessor.load_and_preprocess_image(image_path)
        if img_array is None:
            return {"error": "Could not load image"}
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = self.class_names[predicted_idx]
        
        # Get all class probabilities
        class_probabilities = {
            self.class_names[i]: float(predictions[0][i]) 
            for i in range(len(self.class_names))
        }
        
        # Generate fun features
        confidence_msg = self.get_confidence_message(confidence, predicted_class)
        chef_tip = self.get_chef_tip(predicted_class)
        recipe_url = self.recipe_urls[predicted_class]
        
        # Easter egg: "Guess the Smell" toggle
        smell_msg = "Can't smell pixels yet, working on it. 🔬👃😤"
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "confidence_message": confidence_msg,
            "chef_tip": chef_tip,
            "recipe_url": recipe_url,
            "smell_message": smell_msg,
            "all_probabilities": class_probabilities
        }

def demo_prediction():
    """
    Demo function to test the predictor
    """
    print("🍽️  Food Image Classifier Demo! 🍽️")
    print("=" * 40)
    
    predictor = FoodPredictor()
    
    if predictor.model is None:
        print("❌ Please train the model first by running 'python model/train_model.py'")
        return
    
    # Test with sample images from dataset
    data_dir = 'data'
    if os.path.exists(data_dir):
        for class_name in predictor.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                # Get first image from class
                img_files = [f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    test_image = os.path.join(class_dir, img_files[0])
                    print(f"\n🔍 Testing with {class_name} image...")
                    
                    result = predictor.predict_food(test_image)
                    
                    if "error" not in result:
                        print(f"📊 {result['confidence_message']}")
                        print(f"💡 {result['chef_tip']}")
                        print(f"🔗 Recipe: {result['recipe_url']}")
                        print(f"👃 {result['smell_message']}")
                        print("-" * 40)

if __name__ == "__main__":
    demo_prediction()

# Important Variables:
# - class_names: List of food categories ['pizza', 'pasta', 'salad', 'donuts']
# - class_emojis: Corresponding emojis for visual appeal
# - chef_tips: Dictionary of fun facts and tips for each food class
# - recipe_urls: Hardcoded recipe links for each food type

# Major Functions:
# - get_confidence_message(): Creates fun confidence descriptions
# - get_chef_tip(): Returns random cooking tips
# - predict_food(): Main prediction function with all fun features
# - demo_prediction(): Tests the predictor with sample images
