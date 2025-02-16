# AI Resume Scanner

## Overview
This project is an **AI-powered Resume Scanner** that analyzes resumes and predicts job fields. It also recommends courses based on the predicted job category.

## Implementation History
Initially, we implemented a **TF-IDF + RandomForest Classifier** approach for resume classification. However, the accuracy was not satisfactory, and we decided to implement **BERT (Bidirectional Encoder Representations from Transformers)** for better results.

### 1️⃣ **TF-IDF + RandomForest Approach**
- Used **TF-IDF (Term Frequency - Inverse Document Frequency)** for text vectorization.
- Trained a **RandomForestClassifier** on extracted resume text.
- While it provided a baseline model, its accuracy was limited due to TF-IDF's inability to capture contextual relationships.

### 2️⃣ **BERT-Based Approach (Implementation Attempted)**
- We implemented **BERT** for better contextual understanding of resume text.
- The model was loaded using `transformers` and `torch`, and fine-tuned on labeled resume data.
- However, due to **dependency issues, OpenSSL problems, and conflicts with environment configurations**, the BERT-based model **did not run successfully**.

## Technologies Used
- **Python** (Primary programming language)
- **Streamlit** (For the web app UI)
- **Scikit-learn** (For TF-IDF vectorization and classification)
- **Transformers & PyTorch** (For BERT-based classification, but unresolved issues)
- **PDFMiner** (For extracting text from PDFs)
- **SQLite** (For storing user data and scores)
- **Plotly** (For visualization)
- **Pafy & yt-dlp** (For embedding YouTube career guidance videos)

## Installation
### 1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-repo/AI-Resume-Scanner.git
cd AI-Resume-Scanner
```

### 2️⃣ **Create a Virtual Environment & Install Dependencies**
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3️⃣ **Run the App**
```bash
streamlit run App.py
```

## Next Steps
- Debug and resolve dependency issues to get **BERT-based classification** running.
- Improve resume scoring and recommendations.
- Optimize UI and performance.

## Issues & Troubleshooting
### **Known Issues**
- **BERT Model Not Running**: Encountered OpenSSL, `numpy`, and `torch` issues.
- **pafy YouTube Integration**: Requires `yt-dlp` instead of `youtube-dl`.
- **Environment Conflicts**: Dependencies need proper management with a fresh virtual environment.

### **Workarounds Tried**
- Reinstalled `openssl`, `numpy`, and `torch`.
- Attempted using different `pafy` backends.
- Cleaned and rebuilt virtual environment.

## Contributors
- **Hadi Hijazi**
- **Community Contributions Welcome!**

---
**⚡ Let us know if you face issues or want to contribute!** 🚀
