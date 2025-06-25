"""
Sample Data Generator for Food Image Classifier
Creates placeholder images for testing when you don't have real food images
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_sample_food_images():
    """
    Generate colorful placeholder images for each food class
    This helps you test the classifier without needing real food photos
    """
    
    # Food classes and their representative colors
    food_classes = {
        'pizza': [(255, 100, 100), (255, 200, 100), (100, 255, 100)],  # Red, orange, green
        'pasta': [(255, 220, 150), (255, 180, 100), (200, 150, 100)],  # Pasta colors
        'salad': [(100, 255, 100), (150, 255, 150), (50, 200, 50)],    # Green variations
        'donuts': [(255, 200, 150), (255, 150, 200), (200, 150, 255)]  # Sweet colors
    }
    
    # Create data directory structure
    base_dir = 'data'
    for class_name in food_classes.keys():
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate 50 sample images per class
        for i in range(50):
            # Create a 128x128 image with random food-like patterns
            img = Image.new('RGB', (128, 128), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add some random shapes to simulate food textures
            colors = food_classes[class_name]
            
            # Draw random circles and rectangles
            for _ in range(random.randint(5, 15)):
                color = random.choice(colors)
                shape_type = random.choice(['circle', 'rectangle'])
                
                if shape_type == 'circle':
                    x, y = random.randint(0, 100), random.randint(0, 100)
                    radius = random.randint(10, 30)
                    draw.ellipse([x, y, x+radius, y+radius], fill=color)
                else:
                    x1, y1 = random.randint(0, 80), random.randint(0, 80)
                    x2, y2 = x1 + random.randint(20, 40), y1 + random.randint(20, 40)
                    draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Add class label text
            try:
                draw.text((10, 10), class_name.upper(), fill='black')
            except:
                pass  # Skip if font issues
            
            # Save the image
            img_path = os.path.join(class_dir, f'{class_name}_{i+1:03d}.png')
            img.save(img_path)
    
    print("✅ Sample food images generated successfully!")
    print("📁 Check the 'data' folder for your training images")
    
    # Print summary
    for class_name in food_classes.keys():
        class_dir = os.path.join(base_dir, class_name)
        count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        print(f"   {class_name}: {count} images")

if __name__ == "__main__":
    create_sample_food_images()
