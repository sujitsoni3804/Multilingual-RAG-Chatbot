# 🌐 Multilingual RAG Chatbot

An AI-powered document assistant that lets you chat with your PDF documents in multiple languages. Upload any PDF, ask questions in **English**, **Hindi**, or **Gujarati**, and get accurate answers extracted directly from your document.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Document Processing** | Upload PDFs and extract text using advanced OCR technology |
| 🔍 **RAG-Based Q&A** | Retrieval-Augmented Generation for accurate, context-aware answers |
| 🌍 **Multilingual Support** | Ask questions and get responses in English, Hindi, or Gujarati |
| 🎤 **Voice Input** | Speak your questions using speech-to-text recognition |
| 📊 **Smart Caching** | Previously processed documents are cached for faster access |
| 🔄 **Real-time Streaming** | Watch responses generate in real-time |

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **LLM:** Gemma 3 (4B) via Ollama
- **Embeddings:** LaBSE (Language-agnostic BERT)
- **Vector Search:** FAISS with hybrid search (semantic + TF-IDF)
- **OCR:** Tesseract + PyMuPDF
- **Speech-to-Text:** Whisper (English), IndicWav2Vec (Hindi, Gujarati)
- **Frontend:** HTML, CSS, JavaScript

---

## 📋 Prerequisites

Before running the application, ensure you have:

- **Python 3.10+**
- **Ollama** installed and running ([Download Ollama](https://ollama.com/download))
- **Tesseract OCR** installed ([Installation Guide](https://github.com/tesseract-ocr/tesseract))
- **CUDA** (optional, for GPU acceleration)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sujitsoni3804/Multilingual-RAG-Chatbot.git
cd Multilingual-RAG-Chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

```bash
# Download Gemma model via Ollama
ollama pull gemma3:4b

# Download embedding and speech models
python download_hf_model.py
```

### 5. Run the Application

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`

---

## 📖 How to Use

1. **Upload a PDF** - Click the upload button and select your PDF document
2. **Wait for Processing** - The app extracts text and creates embeddings
3. **Ask Questions** - Type or speak your question in any supported language
4. **Get Answers** - Receive AI-generated responses based on your document

---

## 📁 Project Structure

```
Multilingual-RAG-Chatbot/
├── app.py                   # Main Flask application
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Web interface
├── scripts/
│   ├── gemma_api.py        # LLM integration with Ollama
│   ├── embedding_rag.py    # RAG pipeline and vector search
│   ├── pdf_ocr.py          # PDF text extraction and OCR
│   └── speech_to_text.py   # Voice recognition
├── Models/                  # Downloaded AI models (after setup)
└── PDFs/                    # Sample PDF documents
```

---

