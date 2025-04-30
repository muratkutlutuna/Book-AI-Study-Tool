import os
import logging
import colorlog
import subprocess
import time
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import streamlit as st

# Suppress HuggingFace tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    fmt='%(log_color)s%(levelname)-8s%(reset)s | %(name)s | %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger = colorlog.getLogger('myapp')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# --- Step 0: Ensure Ollama is running and DeepSeek model is loaded ---
def ensure_ollama_running_and_model_loaded():
    try:
        subprocess.run(["ollama", "list"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[+] Ollama is already running.")
    except subprocess.CalledProcessError:
        print("[!] Ollama is not running. Attempting to start it...")
        try:
            subprocess.Popen(["ollama", "serve"])
            time.sleep(5)
            print("[+] Ollama server started.")
        except Exception as e:
            print("[!] Failed to start Ollama server:", e)
            exit(1)

    try:
        subprocess.run(["ollama", "run", "deepseek-llm"], input=b"Hello", timeout=200)
        print("[+] DeepSeek R1 model is ready.")
    except subprocess.TimeoutExpired:
        print("[+] DeepSeek R1 model is loading (may take a while)...")

# --- Step 1: Load and extract PDF text from all files in a folder ---
def extract_texts_from_folder(folder_path):
    all_text = ""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    progress_bar = st.progress(0)
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(folder_path, filename)
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        all_text += f"\n\n--- From {filename} ---\n\n" + text

        progress = int((i + 1) / total_files * 100)
        progress_bar.progress(progress)

    return all_text

# --- Step 2: Chunk the text ---
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    st.write("[+] Chunking text into smaller pieces...")
    progress_bar = st.progress(0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = splitter.create_documents([text])
    for i in range(len(chunks)):
        progress = int((i + 1) / len(chunks) * 100)
        progress_bar.progress(progress)
        time.sleep(0.05)

    st.success("[+] Text chunking completed!")
    return chunks

# --- Step 3: Embed chunks and create vector store ---
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = None

    progress_bar = st.progress(0)
    total_docs = len(docs)
    for i, doc in enumerate(docs):
        if db is None:
            db = FAISS.from_documents([doc], embeddings)
        else:
            db.add_documents([doc])

        progress = int((i + 1) / total_docs * 100)
        progress_bar.progress(progress)

    return db

# --- Step 4: Set up Retrieval-QA chain ---
def setup_qa_chain(db):
    retriever = db.as_retriever(search_type="similarity", k=5)
    llm = OllamaLLM(model="deepseek-r1:70b")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# --- Step 5: Main interaction ---
def ask_question(qa_chain, question):
    st.write("---")
    st.write("🤔 **Question:**")
    st.write(question)
    st.write("---")
    
    progress_bar = st.progress(0)
    thinking_text = st.empty()

    # Get and format the response
    try:
        # First step: Initial thinking
        thinking_text.write("🔄 Analyzing question and retrieving context...")
        progress_bar.progress(5)
        
        # Get response from the QA chain
        response = qa_chain.invoke({
            "query": f"""
            Please think through this step by step:
            1. First, understand the question and its key components
            2. Then, search through the provided context
            3. Finally, formulate a clear and comprehensive answer
            
            Question: {question}
            """
        })
        
        # Extract thinking process and answer
        full_response = response.get("result", "")
        
        # Show thinking process with progress updates
        if "<think>" in full_response:
            # Extract thinking process between <think> tags
            thinking_parts = full_response.split("<think>")[1].split("</think>")[0].strip().split("\n")
            for i, thought in enumerate(thinking_parts):
                if thought.strip():
                    thinking_text.write(f"🤔 {thought.strip()}")
                    progress_bar.progress(5 + (i + 1) * (50 // len(thinking_parts)))
                    time.sleep(0.5)
        
        # Show final answer
        progress_bar.progress(100)
        thinking_text.write("✅ Answer generated successfully!")
        
        # Format the answer text - remove thinking process if present
        answer_text = full_response
        if "</think>" in answer_text:
            answer_text = answer_text.split("</think>")[1].strip()
        
        st.markdown(answer_text)
        st.write("---")
        return answer_text
        
    except Exception as e:
        st.error(f"An error occurred while generating the answer: {str(e)}")
        return None

# --- Step 6: Streamlit UI ---
def run_streamlit_ui():
    st.set_page_config(page_title="AI Book Analyzer", page_icon=":book:", layout="wide")

    # Initialize session state variables
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "db" not in st.session_state:
        st.session_state.db = None
    if "folder_path" not in st.session_state:
        st.session_state.folder_path = "/Users/muratkutlutuna/Documents/ISTQB TAE Kurs"
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
    if "cancel_process" not in st.session_state:
        st.session_state.cancel_process = False

    # Static "Ask something about the books" section
    st.title("AI Book Analyzer using DeepSeek R1 (Ollama)")
    question = st.text_input("Ask something about the books:", placeholder="Ask me some ISTQB Test Automation Engineer exam Questions?")
    cancel_button = st.button("Cancel the process")

    # Handle cancel button
    if cancel_button:
        st.session_state.cancel_process = True
    else:
        st.session_state.cancel_process = False

    # Process the question if submitted
    if question and st.session_state.qa_chain:
        answer = ask_question(st.session_state.qa_chain, question)
        if answer:
            st.write("\n**Answer:**\n", answer)

    # Run setup steps in the background
    if not st.session_state.setup_complete:
        st.write("[+] Setting up the environment in the background...")
        setup_progress = st.progress(0)
        setup_status = st.empty()

        setup_status.write("[+] Checking if Ollama is running...")
        ensure_ollama_running_and_model_loaded()
        setup_progress.progress(20)
        st.success("[✓] Ollama is running successfully!")

        setup_status.write("[+] Extracting text from all PDFs in folder...")
        text = extract_texts_from_folder(st.session_state.folder_path)
        setup_progress.progress(40)
        st.success("[✓] Text extraction completed successfully!")

        setup_status.write("[+] Chunking text into smaller pieces...")
        docs = chunk_text(text)
        setup_progress.progress(60)
        st.success("[✓] Text chunking completed successfully!")

        setup_status.write("[+] Creating vector store...")
        st.session_state.db = create_vector_store(docs)
        setup_progress.progress(80)
        st.success("[✓] Vector store created successfully!")

        setup_status.write("[+] Setting up QA chain...")
        st.session_state.qa_chain = setup_qa_chain(st.session_state.db)
        setup_progress.progress(100)
        st.success("[✓] QA chain setup completed successfully!")

        st.session_state.setup_complete = True
        st.success("Setup complete! You can now ask questions.")

    if not st.session_state.setup_complete:
        st.warning("The setup is still in progress. Please wait...")

if __name__ == "__main__":
    run_streamlit_ui()