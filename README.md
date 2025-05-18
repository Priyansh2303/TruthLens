# TruthLens 🔍 - Deepfake Detection using Deep Learning

TruthLens is a deep learning-based application that detects deepfakes using a pre-trained model. The system analyzes facial features and classifies video frames as real or fake.

---

## 🚀 Features

- Deep learning model using EfficientNet/MesoNet
- Real-time and batch detection modes
- User-friendly interface for selecting video inputs
- High accuracy on benchmark deepfake datasets

---

## 🗂️ Project Structure

```
chat_deepfake/
├── model/                  
├── dataset/               # (You will add this manually)
├── app.py                 # Main app script
├── requirements.txt       # Required Python libraries
└── ...
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Priyansh2303/TruthLens.git
cd TruthLens
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Activate:
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download Required Files

### 🔹 1. Dataset

- Download the dataset ZIP from this Google Drive link:  
  [📁 Dataset Download](https://drive.google.com/drive/folders/1l6KonD5HLDKdkI1iv2733A2F2teZyhhl?usp=sharing)
- Extract it and move the folder to the project directory:

```bash
# Move to project folder
mv your_downloaded_dataset_folder ./dataset/
```


## ▶️ Run the Application

```bash
python app.py
```

---

## 🧠 Model Info

The project uses a pretrained deep learning model trained on public deepfake datasets. You can retrain or replace the model as needed.

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repo and propose improvements.

---


## 🙋‍♂️ Maintainer

**Priyansh Bansal**  
📧 priyanshbansal23march@gmail.com 
🌐 https://github.com/Priyansh2303
