import logging
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
logger = logging.getLogger(__name__)
onnx_model_path = "onnxmodel"

class LiidClassifierModel(object):
    def __init__(self):
        try:
            self.model = ORTModelForSequenceClassification.from_pretrained(onnx_model_path)
            logging.info("The model is loaded")
        except:
            logging.info("Model not loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_labels = np.argmax(logits, axis=1)[0]
        return predicted_labels