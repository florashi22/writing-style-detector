# Writing Style Detector (Author / AI vs Human Stylometry)

This project is a small prototype for detecting differences in writing style across authors.
It extracts simple stylometric features from text (such as sentence length, vocabulary richness,
and punctuation patterns) and trains a machine learning model to classify who wrote a given text.

I built this project to explore how stylometry and data analysis can be used for
academic integrity tools, such as detecting whether essays are consistent with a student's
previous writing style.

## Project Structure

- `data/sample_texts.csv`  
  Example dataset with text samples and author labels.

- `src/preprocess.py`  
  Functions for basic text cleaning and sentence/word splitting.

- `src/features.py`  
  Functions to compute stylometric features from raw text.

- `src/train.py`  
  Script that loads the data, extracts features, trains a classifier, and prints evaluation metrics.

- `src/predict.py`  
  Script that loads the trained model and predicts the most likely author for new input text.

## How to Run

1. Create a virtual environment :

   ```bash
   python -m venv venv
   source venv/bin/activate   
