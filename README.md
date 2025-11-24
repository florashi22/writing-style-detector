# Writing Style Detector  
### Hybrid Stylometry + Embedding-Based AI vs Human Text Classifier

This project is a modern, lightweight writing-style analysis system that combines:

- **Stylometric features** (sentence length, vocabulary richness, punctuation habits)  
- **HuggingFace sentence embeddings**  
- **Machine learning classification**

to distinguish between:

- **Human-written text**  
- **AI-generated text**

The goal is to prototype tools for **academic integrity**, **writer verification**, and **essay style consistency**, similar to what EdTech platforms use to detect suspicious writing patterns.

---

## Features

### Stylometric Feature Extraction
Extracts:
- Vocabulary richness  
- Sentence length  
- Punctuation usage  
- Stopword ratios  
- Word frequency distribution  

### Embedding-Powered Semantic Understanding
Uses the free HuggingFace embedding model 'all-MiniLM-L6-v2' to capture deeper semantic meaning beyond stylometry.

### AI vs Human Classification
Hybrid model combining stylometric + embedding features for robust classification.

### Author Identification 
Prototype classifier for detecting writing differences across multiple authors.

---

## Project Structure
```
Writing Style Detector/
│
├── data/
│ ├── sample_texts.csv # Example author dataset
│ └── ai_vs_human.csv # AI vs Human dataset
│
├── src/
│ ├── preprocess.py # Text cleaning utilities
│ ├── features.py # Stylometric feature extraction
│ ├── embeddings.py # HuggingFace embeddings
│ ├── train.py # Author classifier training
│ ├── train_ai_classifier.py # AI vs Human training
│ ├── predict.py # Author prediction script
│ ├── predict_ai_classifier_emb.py # AI/Human prediction script
│ └── __init__.py
│
├── model.pkl # Author classifier model
├── ai_model_embedded.pkl # AI/Human classifier model
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

## Getting Started

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the AI vs Human classifier
```bash
python -m src.train_ai_classifier
```

### 4. Test predictions (AI vs Human)
```bash
python -m src.predict_ai_classifier_emb
```

### 5. Test author identification (optional)
```bash
python -m src.predict
```