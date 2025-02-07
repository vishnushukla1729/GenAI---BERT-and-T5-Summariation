# Real-Time Summarization using Transformers

## 📌 Overview

This repository contains implementations for real-time text summarization using transformer-based models such as BERT, T5, and GPT. It includes foundational concepts like attention mechanisms and transformer architectures, along with applied implementations for dialogue summarization and API-based text summarization.

## 📂 Project Structure

```
├── BERTandT5.ipynb                     # Introduction to Transformers (PPT + implementation)
├── Dialougue_Summarization_GPT_model.ipynb # GPT-based dialogue summarization with API integration
├── Home.ipynb                           # BERT from scratch + T5 pretrained summarization
├── Source.ipynb                         # Attention mechanism and transformer fundamentals
├── README.md                            # Project documentation
```

## 📜 Detailed File Description

### 1️⃣ BERTandT5.ipynb – Introduction to Transformers & Summarization

- Contains a PPT (presentation) introducing Transformer architectures.
- Covers Self-Attention, Multi-Head Attention, and Feed-Forward Networks.
- Explains BERT and T5 models and their applications in text summarization.
- Includes a practical summarization script using T5 pretrained weights.

#### 🔹 Key Features
- ✔ Understanding Transformer models
- ✔ Using T5 for text summarization
- ✔ Comparison of BERT and T5 architectures

---

### 2️⃣ Dialougue_Summarization_GPT_model.ipynb – GPT-Based Dialogue Summarization

- Implements dialogue summarization using GPT-based models.
- Calls OpenAI’s GPT API to generate and summarize dialogues dynamically.
- Works on structured and unstructured conversation formats.

#### 🔹 Key Features
- ✔ Uses GPT API for text generation and summarization
- ✔ Handles conversational data efficiently
- ✔ Supports integration into chatbots and customer service applications

#### Usage Example:
```python
import openai

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[{"role": "user", "content": "Summarize the following conversation..."}]
)
print(response["choices"][0]["message"]["content"])
```

---

### 3️⃣ Home.ipynb – BERT Implementation from Scratch + T5 Summarization

- Implements BERT from scratch, demonstrating tokenization, embedding, and attention.
- Uses pretrained T5 model for document-level text summarization.
- Compares extractive vs. abstractive summarization.

#### 🔹 Key Features
- ✔ Custom BERT implementation (tokenizer, embeddings, self-attention)
- ✔ Summarization with T5 (pretrained weights)
- ✔ Performance evaluation using ROUGE scores

---

### 4️⃣ Source.ipynb – Attention Mechanism & Transformer Fundamentals

- Breaks down the attention mechanism into step-by-step implementations.
- Covers Scaled Dot-Product Attention, Positional Encoding, and Transformer Architecture.
- Implements key transformer components using PyTorch.

#### 🔹 Key Features
- ✔ Hands-on implementation of Self-Attention
- ✔ Mathematical breakdown of Positional Encoding
- ✔ Step-by-step Transformer Model Building

#### Example: Self-Attention Mechanism
```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1)))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)

# Example usage
Q = K = V = torch.rand(3, 4)
output = self_attention(Q, K, V)
print(output)
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```sh
pip install transformers openai torch fastapi uvicorn
```

### 2️⃣ Running the Transformer Models in Jupyter
Open any notebook (`.ipynb`) in JupyterLab or Google Colab and execute the cells.

### 3️⃣ Running the GPT API for Summarization
- Get an OpenAI API Key and update it in `Dialougue_Summarization_GPT_model.ipynb`.
- Run the notebook to process conversations.

### 4️⃣ Running a Simple FastAPI Server for Summarization
```sh
uvicorn app:main --reload
```
Then send a request:
```sh
curl -X POST "http://127.0.0.1:8000/summarize/" -H "Content-Type: application/json" -d '{"text": "Your long text here"}'
```

---

## 📊 Performance Metrics

- Uses ROUGE Score to evaluate summarization quality.
- Benchmarks GPT vs. T5 for summarization accuracy.
- Compares extractive vs. abstractive summarization.

---

## 📌 Future Improvements

✅ Implement real-time Kafka streaming for large-scale summarization.
✅ Optimize transformer models with ONNX for better inference speed.
✅ Integrate dialogue summarization into chatbots for customer service.
