## Project Submission Report — PolicyAI (Insurance & Loan Document RAG Assistant)

### 1. Problem Statement
Insurance policies and loan agreements contain dense legal wording spread across dozens of pages—coverage limits, waiting periods, exclusions, prepayment penalties, and renewal clauses are difficult for policyholders and borrowers to locate and interpret quickly. Manually reading or searching via plain keywords often misses context across clauses and fails to link answers back to verifiable fine print.

### 2. Solution Overview
**PolicyAI** is a document-grounded Retrieval-Augmented Generation (RAG) workspace built with **Streamlit**, **LangChain**, **Sentence-Transformers**, **FAISS**, and open-source/configurable **LLMs**. Users can sign up or log in, upload text-based insurance policies or loan agreements (PDF), ask questions in plain English, summarize individual documents, or compare two policies side by side—with every claim linked to clickable source citations showing the exact PDF page, detected clause number/title, and extracted text passage.

### 3. Objectives
- **Accurate, Grounded Answers:** Restrict LLM responses strictly to retrieved document passages and explicitly report `"Not specified in the selected documents."` when evidence is absent.
- **Clause- & Page-Level Traceability:** Preserve PDF page numbers and detect clause/section headers during ingestion so every citation maps directly to its original passage.
- **Multi-Document Comparison & Summarization:** Support balanced retrieval across multiple documents for side-by-side policy comparison as well as comprehensive document summaries.
- **Secure Access & Clean UX:** Provide user authentication (Sign Up / Log In / Guest Session) and an animated, responsive workspace with accessible source inspection modals.

---

### 4. System Architecture & RAG Pipeline

```mermaid
flowchart LR
    A["User (Log In / Sign Up / Guest)"] --> B["Upload PDF Policy / Loan"]
    B --> C["Validation & pypdf Page Extraction"]
    C --> D["Clause Header Detection & LangChain Splitter"]
    D --> E["HuggingFace Embeddings (all-MiniLM-L6-v2)"]
    E --> F["FAISS Vector Store (Max Inner Product)"]
    G["User Query / Summary / Compare"] --> F
    F --> H["Top-K Retrieval & Mode Balancing"]
    H --> I["Prompted LLM (HF Local / HF Hub / Gemini / Ollama)"]
    I --> J["JSON & Citation Verifier ([1], [2])"]
    J --> K["Cited Answer + Evidence Panel + Source Modal"]
```

#### Pipeline Stages
1. **Authentication Layer (`core/auth.py`):**
   - Stores registered user accounts in a local SQLite database (`.policyai/users.db`).
   - Hashes passwords with **PBKDF2-HMAC-SHA256** (`200,000` iterations, 16-byte random salt) and verifies credentials with constant-time `hmac.compare_digest`.
   - Supports **Sign Up**, **Log In**, and an ephemeral **Guest Workspace** mode.
2. **Document Ingestion & Clause-Aware Chunking (`core/documents.py`):**
   - Validates PDF header (`%PDF-`), file size (`MAX_FILE_SIZE_MB`, default `20 MB`), encryption status, and extractable text.
   - Extracts text page by page using `pypdf.PdfReader`.
   - Segments pages using regex-based clause/section header detection (`Clause`, `Section`, `Article`, `Condition`, numbered subclauses like `4.2`) and splits text via LangChain `RecursiveCharacterTextSplitter` (`chunk_size=1200`, `chunk_overlap=200`).
   - Converts chunks into LangChain `Document` objects carrying `page`, `clause`, `clause_title`, `chunk_index`, `document`, `document_id`, and `category` metadata.
3. **Dense Embedding & Vector Retrieval (`core/rag.py`):**
   - Generates normalized dense embeddings using `langchain_huggingface.HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2` by default).
   - Indexes per-document chunks in `langchain_community.vectorstores.FAISS` using `DistanceStrategy.MAX_INNER_PRODUCT`.
   - Performs similarity search (`TOP_K=6` default, similarity threshold `0.15` for Q&A/compare and `0.05` for summaries) with round-robin balanced selection across documents in `compare` mode and global score ranking otherwise.
4. **LLM Generation & Citation Verification (`core/rag.py`):**
   - Uses `ChatPromptTemplate` with strict grounding instructions requiring JSON output (`{"text": "...", "source_ids": [...]}`).
   - Supports configurable LLM backends via `.env` (`LLM_PROVIDER`):
     - **Hugging Face Local** (`huggingface_local`, default `Qwen/Qwen2.5-0.5B-Instruct`)
     - **Hugging Face Hub** (`huggingface_hub`, default `Qwen/Qwen2.5-7B-Instruct`)
     - **Google Gemini** (`gemini` via `google-genai`)
     - **Ollama Local** (`ollama`, default `llama3.2`)
   - Parses JSON or inline `[n]` citations via `_parse_rag_response`, verifies source IDs against retrieved passages, and falls back to lexical/score overlap (`_rank_supporting_sources`) when needed.
5. **Frontend Workspace (`ui/` + `app.py`):**
   - Custom HTML/CSS/JS Streamlit component communicating with `app.py` via Streamlit component events.
   - Includes animated **Log In / Sign Up** pages with live password strength meter & inline validation, **Dashboard** with animated SVG RAG flow & progress steps, **Upload Documents** dropzone, **Ask PolicyAI** chat + **Supporting Sources** panel + citation modal, **Compare Policies**, **Document Summaries**, **Document Library**, and **Settings** (with interface motion toggle and `prefers-reduced-motion` support).

---

### 5. Technology Stack

| Layer | Technologies Used |
| --- | --- |
| **Frontend UI** | HTML5, CSS3 (keyframe & SVG animations), Vanilla JavaScript |
| **Web Host & State** | Python 3.12, Streamlit (`streamlit.components.v1`) |
| **Authentication** | SQLite3, `hashlib.pbkdf2_hmac` (SHA-256), `hmac`, `secrets` |
| **PDF Processing** | `pypdf`, LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings & Vector DB** | `sentence-transformers/all-MiniLM-L6-v2`, `langchain-huggingface`, `faiss-cpu` |
| **LLM Orchestration** | `langchain-core`, `transformers`, `huggingface-hub`, `google-genai`, Ollama HTTP API |
| **Testing** | Python `unittest`, `streamlit.testing.v1.AppTest`, Node.js syntax checks |

---

### 6. Project Structure

```text
Insurance-RAG/
├── app.py                  # Streamlit host, session state, and event dispatcher
├── core/
│   ├── auth.py             # SQLite + PBKDF2-HMAC user registration & authentication
│   ├── documents.py        # PDF validation, page extraction, clause detection & chunking
│   └── rag.py              # Embeddings, FAISS retrieval, multi-LLM generation & citations
├── ui/
│   ├── index.html          # Custom Streamlit component entry point
│   ├── style.css           # Responsive styling, auth layout, and UI animations
│   └── app.js              # Client state, form validation, workspace views & modals
├── tests/
│   ├── test_auth.py        # Unit tests for registration, hashing, and login validation
│   ├── test_documents.py   # Unit tests for PDF extraction, clause metadata & error handling
│   ├── test_rag.py         # Unit tests for config, FAISS retrieval, citations & AppTest smoke
│   └── ui-test-fixture.pdf # Sample PDF fixture for UI testing
├── preview.mjs             # Standalone Node.js preview server for UI development
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variable template for LLM & embedding settings
```

---

### 7. Verification & Test Results
All automated unit tests and syntax checks pass:
- **Authentication Tests (`tests/test_auth.py` — 3 tests):** Verified user registration, password hashing exclusion from public profiles, email normalization, duplicate/invalid input rejection, and wrong-credential rejection.
- **Document Processing Tests (`tests/test_documents.py` — 6 tests):** Verified page/clause metadata extraction, multi-page clause carry-over, LangChain `Document` conversion, and rejection of non-PDFs, scan-only/empty PDFs, encrypted PDFs, and oversized files.
- **RAG & App Smoke Tests (`tests/test_rag.py` — 6 tests):** Verified LLM provider resolution, FAISS global similarity ranking, balanced two-document comparison retrieval, JSON/prose citation parsing, end-to-end mocked RAG response generation, and Streamlit `AppTest` smoke execution.
- **Frontend & Python Syntax Checks:** Verified `ui/app.js` and `preview.mjs` with `node --check` and Python modules with `compileall`.

---

### 8. Limitations & Future Scope
- **Scanned PDFs:** Current extraction relies on embedded PDF text (`pypdf`); image-only scanned PDFs require OCR preprocessing before upload.
- **Session-Scoped Documents:** Uploaded PDFs and FAISS indexes live in the active Streamlit session and are cleared on refresh or logout; persistent per-user encrypted document vaults can be added in future versions.
- **Cross-Encoder Reranking & Hybrid Search:** Combining BM25 lexical matching with dense FAISS search and cross-encoder reranking can further improve retrieval precision on numeric policy tables.