# Knowledge Augmented Gen - 📄 A Private Document Summarizer using RAG, LangChain & IBM Watsonx

## 🚀 Objective
This project solves the problem of summarizing and querying **private documents securely and efficiently** using **Retrieval-Augmented Generation (RAG)** architecture.  
Instead of uploading sensitive documents to public LLM services, this solution:
- Stores & indexes documents **locally**.
- Converts them into embeddings for efficient search.
- Retrieves only relevant information chunks.
- Sends **only small, relevant snippets** to the cloud-based **IBM Watsonx.ai LLM** for question answering.

Ideal use-case: Onboarding at a company with a large volume of private policies, guidelines, or project documents you need to quickly understand, but can’t risk uploading publicly.

---

## 🛠️ Technologies Used

| Technology                      | Purpose                                          |
|----------------------------------|--------------------------------------------------|
| **LangChain**                   | Document loading, splitting, RAG pipeline        |
| **Chroma DB**                   | Local vector storage for embeddings              |
| **Hugging Face Embeddings**     | Convert documents to numerical vectors           |
| **IBM Watsonx.ai LLM**          | Language model for summarization and QA          |
| **dotenv**                      | Securely load API keys and configs               |
| **Python 3.8+**                 | Language                                         |

---

## 📂 Project Structure

```
rag_summarizer/
├── data/                         # Private documents
│   └── companyPolicies.txt
├── config/                       # Environment configs
│   └── .env
├── embeddings/                   # Embedding-related functions
│   └── embedder.py
├── retriever/                    # Retriever setup
│   └── retriever.py
├── llm/                          # Watsonx LLM setup
│   └── watsonx_llm.py
├── chain/                        # Building the RAG conversation chain
│   └── qa_chain.py
├── utils/                        # Helpers (optional)
├── main.py                       # Entry point
├── README.md                     # Project documentation
└── requirements.txt              # Dependencies
```

---

## 🔑 Environment Setup

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/Knowledge-Augmented-Gen.git
cd Knowledge-Augmented-Gen
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Setup `.env` File**

Inside `/config/.env`:

```
IBM_URL=your_ibm_url
IBM_API_KEY=your_ibm_api_key
IBM_PROJECT_ID=your_project_id
```

> 📝 *Ensure your IBM Watsonx credentials are secured here.*

---

## 📝 How It Works

### 1. **Indexing Phase:**
- **Load documents**: `.txt` files (company policies, guidelines, manuals, etc.)
- **Split into chunks**: Using `CharacterTextSplitter` for manageable processing.
- **Convert to embeddings**: Using Hugging Face models.
- **Store locally**: In Chroma DB for fast vector-based retrieval.

---

### 2. **Retrieval & QA Phase:**
- Query initiated by user.
- Relevant text chunks retrieved **locally**.
- **Only relevant snippets** are sent to IBM Watsonx.ai LLM.
- Returns accurate, contextual, and private responses.

---

### 3. **Conversation Memory:**
- Retains multi-turn conversation history, allowing context-aware follow-up questions.

---

## 💻 Usage

Run the project:

```bash
python main.py
```

You'll enter an interactive prompt:

```bash
Welcome! Ask your questions (type 'exit' to quit):
Question: What is the mobile policy?
Answer: [LLM-generated answer]

Question: What is the aim of it?
Answer: [Context-aware response]
```

Type `exit` to quit.

---

## 🧩 Customization Ideas

- Replace IBM Watsonx with **Local LLMs** (Mistral, Llama2, GPT4All).
- Expand for **PDF/CSV documents**.
- Add **Flask or Streamlit UI**.
- Integrate **logging and analytics** for usage tracking.

---

## 📌 Why This Project Matters
- Keeps private documents **secure & local**.
- Minimizes data exposure.
- Leverages state-of-the-art **RAG architecture**.
- Easily extensible and production-ready.

---

## 📜 License
MIT License – feel free to fork, modify, and build upon!

---

## ✨ Contributors
- **[Md Yeasin Arafath]** (Developer, Architect)