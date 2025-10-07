"""
Simple inference API for skin cancer classification
Use this in your Flask web application
"""
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


class SkinCancerPredictor:
    """
    Easy-to-use predictor for skin cancer classification

    Usage:
        predictor = SkinCancerPredictor('best_model.pth')
        result = predictor.predict('image.jpg')
        print(result['predicted_class'])
        print(result['confidence'])
    """

    def __init__(self, model_path, device=None):
        """
        Initialize the predictor

        Args:
            model_path: Path to the saved model (.pth file)
            device: 'cuda' or 'cpu'. If None, auto-detect
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Class names
        self.class_names = [
            'Actinic_keratoses',
            'Basal_cell_carcinoma',
            'Benign_keratosis-like_lesions',
            'Dermatofibroma',
            'Melanocytic_nevi',
            'Melanoma',
            'Vascular_lesions'
        ]

        # Friendly class names for display
        self.friendly_names = {
            'Actinic_keratoses': 'Actinic Keratoses',
            'Basal_cell_carcinoma': 'Basal Cell Carcinoma',
            'Benign_keratosis-like_lesions': 'Benign Keratosis-like Lesions',
            'Dermatofibroma': 'Dermatofibroma',
            'Melanocytic_nevi': 'Melanocytic Nevi (Moles)',
            'Melanoma': 'Melanoma',
            'Vascular_lesions': 'Vascular Lesions'
        }

        # Load model
        print("Loading model...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print("Model loaded successfully!")

        # Image transforms
        self.transform = A.Compose([
            A.Resize(384, 384),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def _load_model(self, model_path):
        """Load the trained model"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Recreate exact model architecture from training (model.py)
        class SkinCancerClassifier(torch.nn.Module):
            def __init__(self, model_name, num_classes, dropout=0.3):
                super().__init__()
                self.model_name = model_name
                self.num_classes = num_classes

                # Create base model
                self.model = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,
                    drop_rate=dropout
                )

                # Get feature dimension
                num_features = self.model.num_features

                # Create classifier head (exact copy from model.py)
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(num_features, 1024),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(1024, 512),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(dropout / 2),
                    torch.nn.Linear(512, num_classes)
                )

            def forward(self, x):
                features = self.model(x)
                output = self.classifier(features)
                return output

        # Create model instance
        model = SkinCancerClassifier(
            model_name='vit_large_patch16_384',
            num_classes=len(self.class_names),
            dropout=0.3
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model

    def predict(self, image_path_or_pil):
        """
        Predict the class of a skin lesion image

        Args:
            image_path_or_pil: Either a file path (string) or a PIL Image

        Returns:
            dict: {
                'predicted_class': str,
                'friendly_name': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        # Load image
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')

        # Preprocess
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)

        # Get results
        probs = probabilities.cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Create probability dictionary
        all_probs = {
            self.friendly_names[class_name]: float(prob)
            for class_name, prob in zip(self.class_names, probs)
        }

        return {
            'predicted_class': predicted_class,
            'friendly_name': self.friendly_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': all_probs
        }

    def predict_top_k(self, image_path_or_pil, k=3):
        """
        Get top K predictions

        Args:
            image_path_or_pil: Either a file path (string) or a PIL Image
            k: Number of top predictions to return

        Returns:
            list of dict: Top K predictions with class, friendly_name, and confidence
        """
        result = self.predict(image_path_or_pil)

        # Sort by probability
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Get top K
        top_k = []
        for i in range(min(k, len(sorted_probs))):
            class_friendly, confidence = sorted_probs[i]
            # Find original class name
            class_name = [k for k, v in self.friendly_names.items() if v == class_friendly][0]
            top_k.append({
                'class': class_name,
                'friendly_name': class_friendly,
                'confidence': confidence
            })

        return top_k


# Example usage
if __name__ == '__main__':
    # Initialize predictor
    predictor = SkinCancerPredictor('outputs/vit_large_patch16_384_20251005_080728/best_model.pth')

    # Test prediction
    test_image = 'Skin_Cancer_FullSize/test/Vascular_lesions/ISIC_0031217_jpg.rf.5516ee7043cdd2597e701bd5b2a4a754.jpg'

    # Single prediction
    result = predictor.predict(test_image)
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Predicted: {result['friendly_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll probabilities:")
    for class_name, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name:35s} {prob:.2%}")

    # Top 3 predictions
    print("\n" + "="*60)
    print("TOP 3 PREDICTIONS")
    print("="*60)
    top_3 = predictor.predict_top_k(test_image, k=3)
    for i, pred in enumerate(top_3, 1):
        print(f"{i}. {pred['friendly_name']:35s} {pred['confidence']:.2%}")
