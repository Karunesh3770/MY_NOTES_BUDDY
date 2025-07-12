# MY_NOTES_BUDDY
This project is a Retrieval-Augmented Generation (RAG) pipeline that answers questions from a given PDF using:

- 🧠 LLM: Groq's LLaMA3 (8B)
- 🔎 Retriever: FAISS Vector Index
- 📄 PDF Loader: PDFMiner
- 📈 Embedding Model: HuggingFace `all-mpnet-base-v2`
- 🔗 Framework: LangChain

---

## 🚀 Features

- Load and parse PDF files (page-wise or full)
- Embed text using HuggingFace Sentence Transformers
- Store and search using FAISS vector index
- Retrieve relevant chunks for a given query
- Generate accurate answers using LLaMA3 hosted on Groq

---

## 🧰 Tech Stack

- `Python`
- `LangChain`
- `Groq LLaMA3`
- `FAISS`
- `HuggingFace Transformers`
- `PDFMiner`
- `RecursiveCharacterTextSplitter`

---

## 📦 Installation

```bash
pip install langchain faiss-cpu pdfminer.six langchain_community huggingface_hub
pip install groq
🔐 Environment Variables
python
Copy
Edit
# LangSmith Tracing (optional)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key"

# Groq API Key
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
🧪 Running the System
python
Copy
Edit
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import hub

# Load PDF
loader = PDFMinerLoader("path/to/pdf", mode="single")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)

# Embed text
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Index using FAISS
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# Load RAG prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# Instantiate Groq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

# Set up RetrievalQA
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
    chain_type_kwargs={'prompt': prompt}
)

# Query Loop
while True:
    query = input("Enter your query: ").lower()
    result = rag_chain.run(query)
    print("The answer of query is:", result)
📊 Word Count & Chunk Stats
Total Chunks: len(split_docs)

Total Words: sum(len(doc.page_content.split()) for doc in split_docs)

📎 Sample Output
csharp
Copy
Edit
Enter your query: what is quantum entanglement?
The answer of query is: Quantum entanglement is...
📌 To-Do / Improvements
Add a Streamlit or Flask-based UI

Use advanced re-ranking for retrieved documents

Add support for multi-PDF ingestion

Deploy on cloud (e.g., Hugging Face Spaces or AWS)

🧠 Acknowledgments
LangChain

Groq

FAISS by Facebook AI

HuggingFace Sentence Transformers

