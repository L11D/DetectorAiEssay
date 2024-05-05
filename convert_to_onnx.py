from optimum.onnxruntime import ORTModelForSequenceClassification
model_location = 'model'
model = ORTModelForSequenceClassification.from_pretrained(model_location,
                                                          from_transformers=True)
onnx_model_path = "onnxmodel"
model.save_pretrained(onnx_model_path)
print(f"Модель преобразована в формат ONNX и сохранена в: {onnx_model_path}")
