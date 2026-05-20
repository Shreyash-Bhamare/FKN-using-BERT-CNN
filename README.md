# 🧠 Fake News Detection: BERT + CNN Hybrid Model
 
<div align="center">
**A Data and Vision-Driven Approach to Detecting Misinformation**
 
[Features](#features) • [Results](#results) • [Installation](#installation) • [Usage](#usage) • [Project Structure](#project-structure)
 
</div>
---
 
## 🎯 Overview
 
In the digital era, misinformation spreads faster than ever. **Fake News Detection** is a sophisticated deep learning solution that leverages both **textual and visual cues** from news articles to accurately identify fake news with **98.46% accuracy**.
 
This project combines the power of:
- 🔤 **BERT (Bidirectional Encoder Representations from Transformers)** - for understanding article text
- 🖼️ **CNN (ResNet18)** - for analyzing associated images
- 🔗 **Multimodal Fusion** - for holistic fake news detection
### Why Multimodal?
Real-world fake news often includes **misleading text AND manipulated images**. Single-modality models miss critical signals. Our hybrid approach captures both, achieving superior detection accuracy.
 
---
 
## ✨ Features
 
✅ **Multimodal Analysis** - Analyzes both text content and images simultaneously  
✅ **Real-Time Inference** - Streamlit-based web interface for instant predictions  
✅ **High Accuracy** - 98.46% test accuracy with 0.9985 AUC-ROC score  
✅ **URL-Based Input** - Simply paste a news article URL for instant verification  
✅ **Automatic Content Extraction** - Scrapes title, description, and images from URLs  
✅ **Production-Ready** - Pre-trained model and optimized inference pipeline  
✅ **Comprehensive Evaluation** - Confusion matrix, ROC curves, and detailed metrics  
 
---
 
## 📊 Model Performance
 
| Metric | Real News | Fake News | Overall |
|--------|-----------|-----------|---------|
| **Precision** | 0.9843 | 0.9851 | 0.9847 |
| **Recall** | 0.9879 | 0.9806 | 0.9842 |
| **F1-Score** | 0.9861 | 0.9828 | 0.9844 |
| **Support** | 7,282 | 5,914 | 13,196 |
 
**Overall Accuracy: 98.46%**  
**AUC-ROC Score: 0.9985** (Near-perfect classification)
 
### Confusion Matrix Insights
- ✅ 7,194 Real news correctly classified
- ✅ 5,799 Fake news correctly classified
- ❌ Only 88 False Positives (real classified as fake)
- ❌ Only 115 False Negatives (fake classified as real)
---
 
## 🚀 Quick Start
 
### Prerequisites
- **Python 3.8+**
- **GPU** (Recommended) or CPU
- **8GB RAM** minimum
### Installation
 
1. **Clone the Repository**
```bash
git clone git@github.com-personal:Shreyash-Bhamare/FKN-using-BERT-CNN.git
cd fake-news-detection
```
 
2. **Create Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
 
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
 
### Running the Application
 
#### Option 1: Streamlit Web Interface (Recommended) 🌐
```bash
cd src
streamlit run app.py
```
Then open your browser and navigate to `http://localhost:8501`
 
**How to use:**
1. Paste a news article URL in the input field
2. Click "🚀 Analyze Article"
3. Get instant **Real** or **Fake** prediction
4. View extracted article image
#### Option 2: Python Script Inference
```python
import torch
from transformers import BertTokenizer
from thm import HybridModel
 
# Load model
model = HybridModel()
model.load_state_dict(torch.load('models/hybrid_model_2.pt', map_location='cpu'))
model.eval()
 
# Use for inference
# See evaluate_model.py for complete example
```
 
#### Option 3: Model Evaluation
```bash
cd src
python evaluate_model.py
```
This will generate comprehensive evaluation metrics and visualizations.
 
---
 
## 📋 Dataset Information
 
**EVONs Dataset** - A large-scale multimodal fake news detection dataset
 
- **Total Samples:** 13,196 news articles
- **Real News:** 7,282 (55.2%)
- **Fake News:** 5,914 (44.8%)
- **Modalities:** Text (title + description) + Images
- **Classes:** Binary (Real/Fake)
**Preprocessing Applied:**
- Text lowercasing and punctuation removal
- BERT tokenization with max_length=128
- Image resizing to 224×224 pixels
- Image normalization with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
---
 
## 🧠 Model Architecture
 
```
INPUT
  ├─── TEXT (Title + Description)          IMAGE (Associated Thumbnail)
  │         │                                      │
  │    [BERT Encoder]                      [CNN (ResNet18)]
  │    (768-dim output)                    (Feature Extraction)
  │         │                                      │
  │    [Linear Layer]                      [Feature Projection]
  │    (256-dim)                           (256-dim)
  │         └──────────────┬────────────────┘
  │                        │
  │               [Concatenation Layer]
  │               (512-dim combined features)
  │                        │
  │                  [Dropout (0.5)]
  │                        │
  │               [Fully Connected Layer]
  │               (512 → 128)
  │                        │
  │               [Output Layer]
  │               (128 → 2 classes)
  │                        │
  OUTPUT: [Real / Fake Prediction + Confidence]
```
 
### Key Components
 
| Component | Details |
|-----------|---------|
| **Text Encoder** | BERT (bert-base-uncased) |
| **Vision Encoder** | ResNet18 (pretrained) |
| **Fusion Strategy** | Concatenation |
| **Regularization** | Dropout (0.5), Label Smoothing, Weight Decay |
| **Loss Function** | CrossEntropyLoss with class weights |
| **Optimizer** | AdamW (lr=2e-5) |
| **Training Epochs** | 4 (Early stopping applied) |
 
---
## 💻 System Requirements
 
### Hardware
- **CPU:** Intel i5 / AMD Ryzen 5 (minimum)
- **RAM:** 8GB recommended
- **Storage:** 10GB free space
- **GPU:** NVIDIA RTX 4060 or equivalent (optional but recommended)
### Software
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- pip or conda
---
## 🔍 How It Works
 
### Step-by-Step Process
 
1. **Input:** User provides a news article URL
2. **Scraping:** BeautifulSoup extracts title, description, and main image
3. **Text Processing:** 
   - Combine title + description
   - Tokenize using BERT tokenizer
   - Generate input_ids and attention_mask
4. **Image Processing:**
   - Download and resize to 224×224
   - Normalize pixel values
   - Convert to tensor
5. **Feature Extraction:**
   - BERT processes text → 256-dim embeddings
   - ResNet18 processes image → 256-dim embeddings
6. **Fusion & Classification:**
   - Concatenate features → 512-dim vector
   - Pass through fully connected layers
   - Apply softmax → [Real, Fake] probabilities
7. **Output:** Prediction label with confidence score
---
 
## 🎓 Research Contribution
 
This project addresses key research gaps:
 
- **Multimodal Fusion:** Combines text and image analysis (unlike text-only approaches)
- **BERT + CNN Hybrid:** Leverages transformer power with CNN visual reasoning
- **Real-World Deployment:** Practical Streamlit interface for end-users
- **High Performance:** 98.46% accuracy on EVONs dataset
---
 
## 🚀 Future Enhancements
 
- 🌍 **Multilingual Support:** mBERT, XLM-R for regional languages
- 🔍 **Explainability:** LIME/SHAP integration for interpretable predictions
- 📡 **Real-Time Monitoring:** RSS feeds, social media, trending news APIs
- 🔬 **Evidence Retrieval:** Claim verification with source matching
- 🎨 **Advanced Image Analysis:** Vision Transformers (ViT), CLIP for manipulated images
- 🌐 **Public API & Browser Extension:** Lightweight tools for end-users
- 🤝 **NGO Integration:** Collaboration with fact-checking agencies
---
 
## 🤝 Contributing
 
Contributions are welcome! To contribute:
 
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
---
 
<div align="center">
**Made with ❤️ for combating misinformation**
 
⭐ If you found this project helpful, please consider giving it a star!
 
</div>