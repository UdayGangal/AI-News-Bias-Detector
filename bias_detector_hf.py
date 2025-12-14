from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


class BiasDetectorHF:
    """
    Hybrid Bias Detector:
    - Transformer model detects whether text is biased
    - Keyword logic determines ideological direction
    """

    def __init__(self, model_name="valurank/distilroberta-bias"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.num_labels = self.model.config.num_labels

        self._init_keywords()

    def _init_keywords(self):
        self.left_keywords = [
            "liberal", "equality", "climate", "welfare", "social justice",
            "renewable", "healthcare", "tax the rich", "progressive"
        ]

        self.right_keywords = [
            "conservative", "capitalism", "border", "military", "law and order",
            "national security", "free market", "traditional values"
        ]

    def detect_bias(self, text):
        """
        Main entry point
        """
        try:
            return self._hybrid_detect(text)
        except Exception:
            return self._keyword_only(text)

    # ---------------- CORE HYBRID LOGIC ---------------- #

    def _hybrid_detect(self, text):
        """
        1. Use ML model to detect bias intensity
        2. Use keywords to detect direction
        """

        # --- MODEL PREDICTION ---
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        # Binary model → neutral vs biased
        neutral_prob = probs[0] * 100
        bias_prob = probs[1] * 100

        # --- KEYWORD DIRECTION ---
        text_l = text.lower()
        left_hits = sum(k in text_l for k in self.left_keywords)
        right_hits = sum(k in text_l for k in self.right_keywords)

        # --- DECISION LOGIC ---
        if bias_prob < 55:
            # Model not confident → Neutral
            return {
                "left": 0.0,
                "right": 0.0,
                "neutral": 100.0,
                "overall_bias": "Neutral",
                "confidence": round(neutral_prob, 2),
                "method": "Hybrid (Low Bias Confidence)"
            }

        # Bias exists → decide direction
        if left_hits > right_hits:
            return {
                "left": round(bias_prob, 2),
                "right": 0.0,
                "neutral": round(100 - bias_prob, 2),
                "overall_bias": "Left-Leaning",
                "confidence": round(bias_prob, 2),
                "method": "Hybrid (ML + Keywords)"
            }

        elif right_hits > left_hits:
            return {
                "left": 0.0,
                "right": round(bias_prob, 2),
                "neutral": round(100 - bias_prob, 2),
                "overall_bias": "Right-Leaning",
                "confidence": round(bias_prob, 2),
                "method": "Hybrid (ML + Keywords)"
            }

        else:
            # Biased but direction unclear
            return {
                "left": 0.0,
                "right": 0.0,
                "neutral": round(100 - bias_prob, 2),
                "overall_bias": "Mixed / Unclear",
                "confidence": round(bias_prob, 2),
                "method": "Hybrid (Unclear Direction)"
            }

    # ---------------- FALLBACK ---------------- #

    def _keyword_only(self, text):
        """
        Emergency fallback if model fails
        """
        text = text.lower()
        left_hits = sum(k in text for k in self.left_keywords)
        right_hits = sum(k in text for k in self.right_keywords)

        total = max(left_hits + right_hits, 1)

        left = (left_hits / total) * 100
        right = (right_hits / total) * 100

        if left > right:
            overall = "Left-Leaning"
        elif right > left:
            overall = "Right-Leaning"
        else:
            overall = "Neutral"

        return {
            "left": round(left, 2),
            "right": round(right, 2),
            "neutral": round(100 - max(left, right), 2),
            "overall_bias": overall,
            "confidence": round(max(left, right), 2),
            "method": "Keyword Fallback"
        }
