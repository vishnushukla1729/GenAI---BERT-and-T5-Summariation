✨ Summarization using Bert and SimpleT5 with Greedy Decoding ✨
Added a Pipeline Summarizer for easy understanding how it works
🖋 Author

Vishnu Sujeetkumar Shukla🎓 Graduate Student Researcher, Computational Imaging and Sensing Lab🏫 University of California, Riverside

📌 Overview

This project demonstrates a simple text summarization pipeline using a transformer-based approach. The model follows a T5-like structure and employs greedy decoding to generate concise summaries from input paragraphs.

🚀 Features

✅ Implements a basic encoder-decoder transformer model (SimpleT5).✅ Uses a toy tokenizer for word-level tokenization.✅ Employs greedy decoding for text generation.✅ Compatible with both CPU and GPU environments.

🔧 Installation

Ensure you have Python 3.7+ installed along with PyTorch. Install the necessary dependencies using:

pip install torch

▶️ Usage

Run the script to process an input paragraph and generate a summary:

python summarizer.py

📜 Example Output

📝 Input Paragraph:

"Artificial Intelligence has revolutionized many industries. Deep learning techniques are now widely adopted in tasks such as speech recognition, computer vision, and natural language processing. The rapid development of AI technologies has led to significant breakthroughs and continues to drive innovative solutions."

✨ Generated Summary:

"AI drives innovation with deep learning in speech and vision."

📁 File Structure

.
├── summarizer.py       # Main script implementing summarization
├── README.md           # Documentation

🔮 Future Work

🔹 Train the model on real-world datasets.🔹 Enhance decoding strategies using beam search.🔹 Implement a more sophisticated tokenizer.

📜 License

📝 MIT License


