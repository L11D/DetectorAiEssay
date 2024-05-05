# Detector AI essays
Project about detecting Essays Written by AI. [more](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview)
- directed by `Daniil Lysenko 972202`
## About the project
`LiidClassifierModel` class represents the model used for predicting AI generated essays. 
It is based on the [BERT](https://huggingface.co/google-bert/bert-base-uncased), a transformers model.

## Usage

### CLI

1. Clone this repository to your local machine
```bash
git clone https://github.com/L11D/DetectorAiEssay.git
cd DetectorAiEssay
```
2. Install poetry
```bash
pip install poetry
```
3. Install requirement packages
```bash
poetry install --no-interaction --no-ansi
```
4. Predict
```bash
python main.py predict --file-path /path/to/your/test/dataset.csv
```
5. Predictions will be saved to `./data/results.csv`

### Docker

1. Clone this repository to your local machine
```bash
git clone https://github.com/L11D/DetectorAiEssay.git
cd DetectorAiEssay
```
2. Create docker image
```bash
docker build -t detector_ai_essay:latest .
```
4. Predict
```bash
docker run -v /path/to/your/dataset/folder:/DedectorAiEssay/data \
            detector_ai_essay predict data/your_test_dataset.csv
```
5. Predictions and logs will be saved to `path/to/your/dataset/folder/results.csv`