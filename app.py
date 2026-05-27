import streamlit as st
import tempfile
import os
import time
from fpdf import FPDF
from streamlit_cookies_controller import CookieController

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="InsightRAG Assistant", page_icon="📚", layout="wide")

# Custom CSS for a premium look
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State & Cookies ---
controller = CookieController()
saved_key = controller.get("api_key")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Select your provider, paste your API key, upload a PDF, and ask me anything!"}
    ]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Helpers ---
def generate_chat_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Chat History", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for msg in st.session_state.messages:
        role = "Assistant" if msg["role"] == "assistant" else "You"
        text = f"{role}: {msg['content']}"
        clean_text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=clean_text)
        pdf.ln(5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.unlink(tmp.name)
    return pdf_bytes

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("1. AI Provider")
    provider = st.selectbox("Choose Chat Provider", ["Mistral AI", "OpenAI"])
    
    # API Key Cookie Manager
    api_key_input = st.text_input(f"{provider} API Key", value=saved_key if saved_key else "", type="password")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Key"):
            controller.set("api_key", api_key_input)
            st.success("Saved to Cookies!")
    with col2:
        if st.button("🗑️ Clear Key"):
            controller.remove("api_key")
            st.success("Key cleared!")
    
    # Use the input directly for the app logic
    api_key = api_key_input
    
    st.divider()
    
    st.subheader("2. Upload Document")
    st.info("📝 **Handwritten Notes?** Please scan them using the **Adobe Scan** mobile app first! It automatically converts handwriting into searchable text so this AI can read it perfectly.")
    
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)

    if uploaded_file:
        if st.button("Process Document", type="primary"):
            if not api_key:
                st.error(f"⚠️ Please enter your {provider} API Key first.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_path = tmp_file.name

                status_text.text("Extracting text from PDF...")
                progress_bar.progress(25)
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                status_text.text("Chunking text...")
                progress_bar.progress(50)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = splitter.split_documents(docs)

                status_text.text("Initializing Fast Local Embeddings...")
                progress_bar.progress(50)
                
                # Using Local HuggingFace Embeddings! (Fast and Free)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                status_text.text("Building In-Memory Vector Database... This may take a moment.")
                st.session_state.vectorstore = Chroma(embedding_function=embeddings)
                
                total_chunks = len(chunks)
                batch_size = 50
                start_time = time.time()
                
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i+batch_size]
                    st.session_state.vectorstore.add_documents(batch)
                    
                    chunks_processed = min(i + len(batch), total_chunks)
                    percent_complete = int((chunks_processed / total_chunks) * 100)
                    
                    elapsed_time = time.time() - start_time
                    time_per_chunk = elapsed_time / chunks_processed
                    chunks_remaining = total_chunks - chunks_processed
                    eta_seconds = int(time_per_chunk * chunks_remaining)
                    
                    # Ensure percent is between 0 and 100 for st.progress
                    safe_percent = max(0, min(100, percent_complete))
                    
                    # Calculate percentage within the remaining 50% of the bar (50 to 100)
                    bar_progress = 50 + int(safe_percent / 2)
                    
                    progress_bar.progress(bar_progress)
                    status_text.text(f"Building Database: {safe_percent}% Complete | ETA: ~{eta_seconds} seconds remaining")
                
                progress_bar.progress(100)
                status_text.text("Processing Complete!")
                st.success("Vector database created in-memory! You can now chat.")

    st.divider()
    if len(st.session_state.messages) > 1:
        pdf_bytes = generate_chat_pdf()
        st.download_button(
            label="📥 Download Chat as PDF",
            data=pdf_bytes,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )
        st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Select your provider, paste your API key, upload a PDF, and ask me anything!"}
        ]
        st.rerun()

# --- Main Chat Interface ---
st.title("📚 InsightRAG Assistant")

st.warning("⚠️ **PRO TIP FOR BEST RESULTS:** This AI acts as a highly-targeted search engine, not a summarizer. Please **do not ask broad questions** like *'Summarize this entire book'*. Instead, ask **specific, targeted questions** (e.g., *'What are the limitations of the NNJAG model?'* or *'What does page 12 say about X?'*).")
st.error("🚨 **CRITICAL:** This app uses a lightning-fast Temporary Memory Database. If you refresh or close this tab, your entire chat and PDF will be permanently deleted. If your chat is important, make sure to use the **📥 Download Chat as PDF** button in the sidebar before leaving!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about your document...")

if query:
    if not api_key:
        st.error("⚠️ Please enter an API Key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.error("⚠️ Please upload and process a PDF in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
                    )

                    if provider == "Mistral AI":
                        llm = ChatMistralAI(model="mistral-small-2506", mistral_api_key=api_key)
                    else:
                        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

                    contextualize_q_system_prompt = """Given a chat history and the latest user question \
                    which might reference context in the chat history, formulate a standalone question \
                    which can be understood without the chat history. Do NOT answer the question, \
                    just reformulate it if needed and otherwise return it as is."""
                    
                    contextualize_q_prompt = ChatPromptTemplate.from_messages([
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    
                    history_aware_retriever = create_history_aware_retriever(
                        llm, retriever, contextualize_q_prompt
                    )

                    qa_system_prompt = """You are a helpful AI assistant. \
                    Use ONLY the following pieces of retrieved context to answer the question. \
                    If the answer is not present in the context, say "I could not find the answer in the document." \
                    
                    {context}"""
                    
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", qa_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                    response = rag_chain.invoke({
                        "input": query,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response["answer"]
                    
                    source_docs = response.get("context", [])
                    sources_set = set()
                    for doc in source_docs:
                        page = doc.metadata.get("page", "Unknown")
                        sources_set.add(f"Page {page}")
                    
                    if sources_set:
                        answer += f"\n\n**Sources:** {', '.join(sorted(sources_set))}"

                    st.write_stream(stream_text(answer))
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.append(HumanMessage(content=query))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")