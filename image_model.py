"""
Image Classification Model for Skin Disease Detection
ResNet50-based model for visual skin condition analysis
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import logging

class CNNResNet(nn.Module):
    """ResNet50 model for skin disease classification"""
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True            
            
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class SkinDiseaseClassifier:
    """Wrapper class for skin disease image classification"""
    
    def __init__(self, model_path="models/skin_model.pth"):
        self.model_path = model_path
        self.model = None
        
        # Actual class names from the trained model (from ImageFolder dataset)
        self.class_names = [
            'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous',
            'Candidiasis', 'DrugEruption', 'Eczema', 'Infestations_Bites',
            'Lichen', 'Lupus', 'Moles', 'Psoriasis',
            'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage',
            'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 'Vasculitis',
            'Vitiligo', 'Warts'
        ]  # 22 classes matching the trained model
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _format_class_name(self, class_name):
        """Convert class name to user-friendly format"""
        # Replace underscores with spaces and capitalize
        formatted = class_name.replace('_', ' ')
        return formatted
        
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            # Initialize model with correct number of classes
            num_classes = len(self.class_names)
            self.model = CNNResNet(num_classes=num_classes)
            
            # Load the state dict
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logging.info(f"Skin disease classification model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading skin disease model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess uploaded image for model input"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return image_tensor.to(self.device)
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image):
        """
        Predict skin disease from uploaded image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Prediction results with class probabilities and top prediction
        """
        try:
            if self.model is None:
                if not self.load_model():
                    return None
            
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            if image_tensor is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                # Get top 3 predictions
                top_probs, top_classes = torch.topk(probabilities, 3, dim=1)
                
                results = {
                    'predicted_class': self._format_class_name(self.class_names[predicted_class.item()]),
                    'predicted_class_raw': self.class_names[predicted_class.item()],
                    'confidence': confidence.item(),
                    'top_predictions': [
                        {
                            'class': self._format_class_name(self.class_names[top_classes[0][i].item()]),
                            'class_raw': self.class_names[top_classes[0][i].item()],
                            'probability': top_probs[0][i].item()
                        }
                        for i in range(3)
                    ],
                    'all_probabilities': {
                        self._format_class_name(self.class_names[i]): probabilities[0][i].item() 
                        for i in range(len(self.class_names))
                    }
                }
                
                return results
                
        except Exception as e:
            logging.error(f"Error during image prediction: {str(e)}")
            return None
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'model_type': 'ResNet50 Transfer Learning',
            'input_size': '128x128 RGB',
            'num_classes': len(self.class_names),
            'classes': self.class_names,
            'device': str(self.device)
        }
