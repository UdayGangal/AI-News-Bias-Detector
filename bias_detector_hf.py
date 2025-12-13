from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


class BiasDetectorHF:
    def __init__(self, model_name="valurank/distilroberta-bias"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.num_labels = self.model.config.num_labels
        self.id2label = self.model.config.id2label

        self._init_keywords()

    def _init_keywords(self):
        self.left_keywords = ['liberal', 'equality', 'climate', 'welfare']
        self.right_keywords = ['conservative', 'capitalism', 'border', 'military']
        self.neutral_keywords = ['report', 'data', 'officials', 'according to']

    def detect_bias(self, text):
        try:
            return self._detect_with_model(text)
        except Exception:
            return self._detect_with_keywords(text)

    def _detect_with_model(self, text):
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

        # SAFE default
        scores = {"left": 0.0, "neutral": 0.0, "right": 0.0}

        if self.num_labels == 2:
            # neutral vs biased
            scores["neutral"] = probs[0] * 100
            scores["right"] = probs[1] * 100

        elif self.num_labels == 3:
            for i, label in self.id2label.items():
                label = label.lower()
                if "left" in label:
                    scores["left"] = probs[i] * 100
                elif "right" in label or "conservative" in label:
                    scores["right"] = probs[i] * 100
                else:
                    scores["neutral"] = probs[i] * 100

        max_class = max(scores, key=scores.get)

        return {
            "left": round(scores["left"], 2),
            "right": round(scores["right"], 2),
            "neutral": round(scores["neutral"], 2),
            "overall_bias": max_class.capitalize(),
            "method": "Hugging Face Model"
        }

    def _detect_with_keywords(self, text):
        text = text.lower()

        l = sum(k in text for k in self.left_keywords)
        r = sum(k in text for k in self.right_keywords)
        n = sum(k in text for k in self.neutral_keywords)

        total = max(l + r + n, 1)

        scores = {
            "left": (l / total) * 100,
            "neutral": (n / total) * 100,
            "right": (r / total) * 100
        }

        max_class = max(scores, key=scores.get)

        return {
            "left": round(scores["left"], 2),
            "right": round(scores["right"], 2),
            "neutral": round(scores["neutral"], 2),
            "overall_bias": max_class.capitalize(),
            "method": "Keyword Fallback"
        }
