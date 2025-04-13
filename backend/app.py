from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Define the FundusPreprocess class
class FundusPreprocess:
    def __init__(self, clip_limit=2.0, grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        
    def __call__(self, img):
        # Convert to LAB color space
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge channels and convert back to RGB
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        # Gamma correction
        gamma = 1.0  # Fixed gamma for validation
        img_gamma = np.power(img_clahe/255.0, gamma) * 255.0
        
        return Image.fromarray(img_gamma.astype('uint8'))

# Load your model
def load_model(path='mobilenet_fundus_gamedfash552.pth'):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = EnhancedMobileNetV3(num_classes=len(checkpoint['classes']), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.classes = checkpoint['classes']
    model.input_size = checkpoint['input_size']
    return model

# Define the EnhancedMobileNetV3 class
class EnhancedMobileNetV3(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=pretrained)
        
        # Freeze all layers initially
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze later layers for fine-tuning
        for param in self.base_model.features[-8:].parameters():
            param.requires_grad = True
            
        # ROI attention mechanism
        self.roi_attention = torch.nn.Sequential(
            torch.nn.Conv2d(960, 256, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1, 1),
            torch.nn.Sigmoid()
        )
        
        # Enhanced classifier
        original_in_features = self.base_model.classifier[-1].in_features
        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(original_in_features, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
        
        # Projection layer
        self.feature_projection = torch.nn.Conv2d(960, original_in_features, kernel_size=1)
        
    def forward(self, x):
        features = self.base_model.features(x)
        
        # ROI attention
        attention_weights = self.roi_attention(features)
        attended_features = features * attention_weights
        
        # Project features
        projected_features = self.feature_projection(attended_features)
        
        # Pooling and classification
        pooled = torch.nn.functional.adaptive_avg_pool2d(projected_features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        return self.base_model.classifier(flattened)

# Initialize the model and preprocessing
model = load_model()
model.eval()

# Define image transformations
medical_preprocess = FundusPreprocess()
transform = transforms.Compose([
    transforms.Lambda(lambda x: medical_preprocess(x)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get base64 image from request
        data = request.json
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Return results
        result = {
            'prediction': 'Glaucoma' if prediction == 1 else 'Normal',
            'confidence': float(confidence * 100),
            'probability': float(probabilities[0][1].item() * 100)  # Probability of glaucoma
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 