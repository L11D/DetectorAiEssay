from optimum.onnxruntime import ORTModelForSequenceClassification
model_location = 'results/checkpoint-10'
model = ORTModelForSequenceClassification.from_pretrained(model_location,
                                                          from_transformers=True)
onnx_model_path = "model.onnx"
model.save_pretrained(onnx_model_path)
print(f"Модель преобразована в формат ONNX и сохранена в: {onnx_model_path}")
