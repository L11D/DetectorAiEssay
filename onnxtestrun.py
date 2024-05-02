from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np

onnx_model_path = "model.onnx"
ort_model = ORTModelForSequenceClassification.from_pretrained(onnx_model_path)
tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)

def predict(text, tokenizer, ort_model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = ort_model(**inputs)
    logits = outputs.logits
    predicted_labels = np.argmax(logits, axis=1)[0]
    return predicted_labels

sample_text = "This is a sample text."


# competition_dataset_path = 'dataset/train_essays.csv'
# competition_dataset = pd.read_csv(competition_dataset_path)
# competition_dataset['label'] = competition_dataset['generated']
# competition_dataset = competition_dataset.drop(columns=['prompt_id', 'id', 'generated'])
# sample_text = competition_dataset.loc[0]['text']

predicted_label = predict(sample_text, tokenizer, ort_model)
print(f"Predicted label: {predicted_label}")