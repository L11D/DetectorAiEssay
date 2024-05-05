import logging
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import sys
import os
from datasets import Dataset

logger = logging.getLogger(__name__)
onnx_model_path = "onnxmodel"
save_path = "data/results.csv"

class LiidClassifierModel(object):
    """
    Model for classifying AI generated essay based on BERT model
    """
    def __init__(self):
        try:
            self.model = ORTModelForSequenceClassification.from_pretrained(onnx_model_path)

            logging.info("The model is loaded")
        except Exception as e:
            logger.exception("Failed to load the model", exc_info=True)
            print("An error occurred while loading the model.")
            sys.exit(1)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)
            logging.info("The tokenizer is loaded")
        except Exception as e:
            logger.exception("Failed to load the tokenizer", exc_info=True)
            print("An error occurred while loading the tokenizer.")
            sys.exit(1)

    def __preprocess_data(self, df):
        """
        Preprocess the input dataframe with tokenizer
        :param df:
        :return:
        """
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        logger.info("Preprocessing data")
        df = Dataset.from_pandas(df)
        eval_dataset = df.map(preprocess_function, batched=True)
        inputs = {
            "input_ids": np.array(eval_dataset['input_ids']).astype(np.int64),
            "attention_mask": np.array(eval_dataset['attention_mask']).astype(np.int64),
            "token_type_ids": np.array(eval_dataset['token_type_ids']).astype(np.int64)
        }
        logger.info("Preprocessing data completed")
        return inputs

    def predict(self, file_path):
        """
        Predict from data in file_path and save results data
        :param file_path:
        :param out_path:
        :return:
        """
        try:
            eval_df = pd.read_csv(file_path)
            logging.info("Dataframe loaded")
        except Exception as e:
            logger.exception("Failed to load data", exc_info=True)
            print("An error occurred while loading data.")
            sys.exit(1)

        eval_dataset = self.__preprocess_data(eval_df)
        logger.info("Predicting data")
        outputs = self.model(**eval_dataset)
        logger.info("Predicting data completed")
        predicted_labels = np.argmax(outputs.logits, axis=1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        out = pd.DataFrame(data={'id': eval_df.id, 'generated': predicted_labels.tolist()})
        out.to_csv(save_path, index=False)
        logger.info(f'predictions saved to {save_path}')
        return f'predictions saved to {save_path}'
