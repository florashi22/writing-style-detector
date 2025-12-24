# AI-Enhanced Authorship & Integrity Intelligence System  
### Hybrid Stylometry + Embedding-Based AI vs Human Text Classifier
## Problem Statement
As generative AI tools become increasingly integrated into content creation workflows, enterprises face growing challenges in authorship attribution, academic integrity, and content governance. Traditional rule-based or single-model approaches struggle to balance accuracy, interpretability, and scalability.

This project explores how hybrid AI systems—combining interpretable stylometric features with semantic embedding-based machine learning models—can provide robust, explainable, and scalable authorship intelligence for real-world enterprise use cases.

## Product Vision
This system is designed as a modular AI intelligence layer that can be embedded into enterprise platforms to:
Detect AI-generated vs human-written content
Identify distinctive writing-style signatures
Support integrity, compliance, and trust-focused workflows
Enable data-driven product decisions around AI governance

## Enterprise & Product Use Cases

#### Academic Integrity Platforms
Flag anomalous writing patterns to assist educators and institutions in integrity review workflows.
#### Enterprise Content Governance
Monitor internal documentation and externally published content for authorship consistency and compliance.
#### Publishing & Media Analytics
Detect deviations from established brand voice or editorial standards.
#### AI Trust & Safety Tooling
Support governance frameworks that assess the responsible use of generative AI across organizations.

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

## Example Output
{
  "authorship": "Human",
  "confidence_score": 0.924,
  "style_signatures": {
    "avg_sentence_length": 14.2,
    "lexical_diversity": 0.76,
    "punctuation_density": 0.11
  }
}

## Interpretation:
The model predicts the text is human-written with high confidence, supported by consistent stylometric and semantic signals. These outputs are designed to support human-in-the-loop decision systems, not black-box automation.

## Limitations & Product Roadmap
#### Current Limitations
Focuses on static stylometric features
English-language datasets only
Offline inference (non-real-time)

#### Future Directions
Integrate LLM-based style reasoning for deeper generative attribution
Deploy as containerized microservices for enterprise scalability
Extend to multilingual embeddings and cross-domain writing analysis
Build dashboard-level visualizations for non-technical stakeholders
