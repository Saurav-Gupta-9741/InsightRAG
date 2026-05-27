# 📚 InsightRAG Assistant

InsightRAG Assistant is a lightning-fast, premium Retrieval-Augmented Generation (RAG) web application built with Streamlit and LangChain. It allows you to upload massive PDF documents and instantly chat with them using either Mistral AI or OpenAI's powerful language models.

---

## 🎯 The Journey: What We Built and Why

When we first envisioned this project, we wanted to build a RAG system with a permanent database where users could upload gigabytes of PDFs and store them forever. 

However, we realized a critical limitation: deploying permanent databases (like ChromaDB or Pinecone) on free-tier cloud providers is heavily restricted, expensive, and prone to data loss when free servers go to sleep.

**Our Solution:** We completely abandoned the persistent storage model and built a highly optimized **Temporary In-Memory Architecture**. 
*   **The Benefit:** The application is completely stateless, meaning it can be deployed on 100% free hosting (like Streamlit Community Cloud) with zero maintenance. 
*   **The Speed:** Because everything lives in RAM, processing is instantaneous. 
*   **Data Retention:** To solve the data-loss issue, we built a custom "Export Chat to PDF" engine using `fpdf2`, allowing users to manually download and save their important conversations before the temporary database vanishes.

---

## 🔑 Key Features

- **Bring Your Own Key (BYOK):** Multi-provider support! Bring either your Mistral AI or OpenAI API key.
- **Browser Memory:** The app uses `streamlit-cookies-controller` to securely cache your API key in your browser. You never have to re-paste it when you refresh!
- **100% Free Local Embeddings:** We replaced expensive API-based embeddings with **HuggingFace Local Embeddings** (`all-MiniLM-L6-v2`). This bypasses API rate limits entirely and processes massive 400-page books using the server's CPU for $0.00.
- **Visual ETA Progress Tracker:** A custom-built progress bar tracks the embedding progress and calculates the Estimated Time of Arrival (ETA) in real-time.
- **History-Aware Chat:** The LLM remembers context. If you ask a follow-up question, it intelligently reformulates the query before searching the PDF.

---

## 🧠 Glossary (For Non-Technical Readers)

If you are a recruiter or someone exploring AI, here is how the underlying technology works:

*   **RAG (Retrieval-Augmented Generation):** A framework that gives an AI "open book" access to your documents. Instead of relying on its pre-trained knowledge, RAG forces the AI to search your specific PDF to find the answer.
*   **Chunking:** AI models can't read a 400-page book all at once. "Chunking" is the process of slicing the PDF into tiny 1000-character paragraphs so they are easy to search.
*   **Embeddings:** The process of turning text into complex mathematical coordinates (vectors). For example, the words "dog" and "puppy" will have very similar mathematical coordinates.
*   **Vector Database:** A special database designed to hold these mathematical coordinates. When you ask a question, the database finds the chunks of text that have the most similar "coordinates" to your question.

---

## 💻 How to Install & Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/InsightRAG.git
   cd InsightRAG
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit server:**
   ```bash
   python -m streamlit run app.py
   ```

4. **Open your browser:** Navigate to `http://localhost:8501` and start chatting!

---

## 📝 A Note on Handwritten Notes

If you want the AI to read handwritten notes, you must first scan them using an app like **Adobe Scan** or **Google Drive**. These mobile apps automatically run OCR (Optical Character Recognition) on your phone and embed invisible, searchable text into the PDF. Without this, the AI will only see a blank picture and fail to process the document.
