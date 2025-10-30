"""
Bias Detector with Hugging Face Pre-trained Model
Uses transformer-based model for accurate political bias classification
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class BiasDetectorHF:
    def __init__(self, model_name="valurank/distilroberta-bias"):
        """
        Initialize with Hugging Face pre-trained model
        Default model: valurank/distilroberta-bias
        This model is trained on political news articles and classifies text as:
        - Left-leaning
        - Center/Neutral
        - Right-leaning
        """
        print(f"   Loading model: {model_name}")
        
        # Initialize keyword detector as fallback (always available)
        self._init_keyword_detector()
        
        try:
            # Determine device (GPU if available, else CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"   Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Get number of labels from model config
            self.num_labels = self.model.config.num_labels
            print(f"   Model has {self.num_labels} output labels")
            
            # Get label names if available
            if hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
                print(f"   Labels: {self.id2label}")
            else:
                # Default mapping based on common patterns
                if self.num_labels == 2:
                    self.id2label = {0: 'neutral', 1: 'biased'}
                elif self.num_labels == 3:
                    self.id2label = {0: 'left', 1: 'neutral', 2: 'right'}
                else:
                    self.id2label = {i: f'class_{i}' for i in range(self.num_labels)}
            
            print(f"   ✅ Model loaded successfully!")
            self.model_loaded = True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {str(e)}")
            print(f"   Falling back to keyword-based detection...")
            self.model_loaded = False
    
    def _init_keyword_detector(self):
        """Initialize keyword-based detector (always available as fallback)"""
        self.left_keywords = [
            'progressive', 'liberal', 'equality', 'social justice', 'diversity',
            'climate change', 'renewable energy', 'healthcare for all', 'universal',
            'regulation', 'welfare', 'immigration reform', 'gun control',
            'minimum wage', 'workers rights', 'union', 'tax the rich',
            'systemic racism', 'lgbtq', 'reproductive rights', 'environment',
            'sustainability', 'public education', 'affordable housing',
            'medicare', 'medicaid', 'green new deal', 'racial justice'
        ]
        
        self.right_keywords = [
            'conservative', 'traditional values', 'free market', 'capitalism',
            'deregulation', 'lower taxes', 'small government', 'second amendment',
            'law and order', 'border security', 'illegal immigration',
            'pro-life', 'family values', 'religious freedom', 'patriot',
            'national security', 'military strength', 'fiscal responsibility',
            'individual liberty', 'limited government', 'constitutional rights',
            'school choice', 'energy independence', 'law enforcement'
        ]
        
        self.neutral_keywords = [
            'bipartisan', 'compromise', 'moderate', 'balanced', 'objective',
            'evidence-based', 'factual', 'data shows', 'research indicates',
            'experts say', 'according to', 'both sides', 'nonpartisan',
            'analysis', 'study', 'report', 'officials'
        ]
    
    def detect_bias(self, text):
        """
        Detect political bias in text
        Returns: dict with bias percentages and overall lean
        """
        if self.model_loaded:
            return self._detect_with_model(text)
        else:
            return self._detect_with_keywords(text)
    
    def _detect_with_model(self, text):
        """Use Hugging Face pre-trained model for bias detection"""
        try:
            # Tokenize input text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert logits to probabilities using softmax
            probabilities = F.softmax(logits, dim=-1)
            probs = probabilities[0].cpu().numpy()
            
            # Map probabilities based on number of labels
            if self.num_labels == 2:
                # Binary classification: neutral vs biased
                # Map to our 3-class format
                bias_scores = {
                    'left': 0.0,
                    'neutral': float(probs[0]) * 100,
                    'right': float(probs[1]) * 100
                }
            elif self.num_labels == 3:
                # Three-class classification
                # Try to map labels intelligently
                label_0 = self.id2label[0].lower()
                label_1 = self.id2label[1].lower()
                label_2 = self.id2label[2].lower()
                
                # Determine which label corresponds to what
                if 'left' in label_0 or 'liberal' in label_0:
                    bias_scores = {
                        'left': float(probs[0]) * 100,
                        'neutral': float(probs[1]) * 100,
                        'right': float(probs[2]) * 100
                    }
                elif 'right' in label_0 or 'conservative' in label_0:
                    bias_scores = {
                        'right': float(probs[0]) * 100,
                        'neutral': float(probs[1]) * 100,
                        'left': float(probs[2]) * 100
                    }
                else:
                    # Default mapping
                    bias_scores = {
                        'left': float(probs[0]) * 100,
                        'neutral': float(probs[1]) * 100,
                        'right': float(probs[2]) * 100
                    }
            else:
                # More than 3 labels - use keyword fallback
                print(f"   ⚠️  Model has {self.num_labels} labels, using keyword detection")
                return self._detect_with_keywords(text)
            
            # Determine overall bias
            max_category = max(bias_scores.items(), key=lambda x: x[1])
            
            if max_category[1] > 50:
                if max_category[0] == 'left':
                    overall_bias = 'Left-Leaning'
                elif max_category[0] == 'right':
                    overall_bias = 'Right-Leaning'
                else:
                    overall_bias = 'Neutral/Balanced'
            elif max_category[1] > 40:
                if max_category[0] == 'left':
                    overall_bias = 'Slightly Left-Leaning'
                elif max_category[0] == 'right':
                    overall_bias = 'Slightly Right-Leaning'
                else:
                    overall_bias = 'Mostly Neutral'
            else:
                overall_bias = 'Mixed/Unclear'
            
            return {
                'left': round(bias_scores['left'], 2),
                'right': round(bias_scores['right'], 2),
                'neutral': round(bias_scores['neutral'], 2),
                'overall_bias': overall_bias,
                'method': 'Hugging Face ML Model',
                'confidence': round(max_category[1], 2)
            }
            
        except Exception as e:
            print(f"   ⚠️  Model prediction error: {str(e)}")
            print(f"   Switching to keyword-based detection...")
            return self._detect_with_keywords(text)
    
    def _detect_with_keywords(self, text):
        """Fallback keyword-based detection"""
        text_lower = text.lower()
        
        left_count = sum(1 for kw in self.left_keywords if kw in text_lower)
        right_count = sum(1 for kw in self.right_keywords if kw in text_lower)
        neutral_count = sum(1 for kw in self.neutral_keywords if kw in text_lower)
        
        total_count = left_count + right_count + neutral_count
        
        if total_count == 0:
            return {
                'left': 33.33,
                'right': 33.33,
                'neutral': 33.34,
                'overall_bias': 'Neutral (No clear indicators)',
                'method': 'Keyword-based (Fallback)',
                'confidence': 33.34
            }
        
        left_percent = (left_count / total_count) * 100
        right_percent = (right_count / total_count) * 100
        neutral_percent = (neutral_count / total_count) * 100
        
        max_percent = max(left_percent, right_percent, neutral_percent)
        
        if max_percent == left_percent and left_percent > 40:
            overall_bias = 'Left-Leaning'
        elif max_percent == right_percent and right_percent > 40:
            overall_bias = 'Right-Leaning'
        elif neutral_percent > 40:
            overall_bias = 'Neutral/Balanced'
        else:
            overall_bias = 'Mixed/Unclear'
        
        return {
            'left': round(left_percent, 2),
            'right': round(right_percent, 2),
            'neutral': round(neutral_percent, 2),
            'overall_bias': overall_bias,
            'method': 'Keyword-based (Fallback)',
            'confidence': round(max_percent, 2)
        }