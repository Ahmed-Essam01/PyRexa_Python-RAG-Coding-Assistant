# PyRexa: Python RAG Coding Assistant

A conversational **Retrieval-Augmented Generation (RAG)** assistant for Python questions. PyRexa retrieves similar Stack Overflow-style Q&A pairs using **hybrid search** (BM25 keyword search + dense vector search), then passes them as context to **Google Gemini** to generate a tailored, runnable answer, with short-term conversation memory.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)
![Colab](https://img.shields.io/badge/Runs%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

---

## Features

- **Hybrid retrieval**: combines BM25 (sparse / keyword) and ChromaDB (dense / semantic) search, merged with Reciprocal Rank Fusion via LangChain's `EnsembleRetriever`.
- **Large knowledge base**: 100,000 sampled Q&A pairs from the [`luisroque/instruct-python-500k`](https://huggingface.co/datasets/luisroque/instruct-python-500k) dataset (~501k rows in total).
- **Text cleaning pipeline**: HTML unescaping, tag stripping, whitespace normalization, and filtering of very short entries.
- **Context-aware answers**: the prompt instructs the LLM to *adapt* retrieved answers to the user's exact question, explain partial matches, or fall back to general Python knowledge when the context isn't relevant.
- **Conversational memory**: a custom sliding-window chat history (`Last5ChatMessageHistory`) keeps the most recent messages per session.
- **Interactive CLI loop**: chat with the assistant directly in the notebook.

---

## Architecture

### Indexing pipeline

```
Raw Dataset (instruct-python-500k)
            │
            ▼
Sampling Subset (100,000 rows)
            │
            ▼
Text Preprocessing (HTML unescape + regex cleanup)
            │
            ▼
LangChain Documents
(page_content = question, metadata = answer)
            │
     ┌──────┴───────────────┐
     ▼                      ▼
ChromaDB Vector Store    BM25 Index
(all-MiniLM-L6-v2)       (tokenized terms)
     │                      │
     ▼                      ▼
Dense Retriever (k=10)   Sparse Retriever (k=10)
     └──────────┬───────────┘
                ▼
   EnsembleRetriever (RRF, weights 0.5 / 0.5)
```

### Query pipeline

```
User input + session_id
        │
        ▼
RunnableWithMessageHistory ──► load chat history (last 10 messages)
        │
        ▼
Hybrid search (BM25 + Chroma) ──► fused results
        │
        ▼
Format context  ([1] Q: ... A: ...)
        │
        ▼
Prompt = system rules + retrieved context + chat history + user question
        │
        ▼
Google Gemini (ChatGoogleGenerativeAI)
        │
        ▼
Answer  ──► saved back to chat history
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 (`rank_bm25`) |
| LLM | Google Gemini via `langchain-google-genai` |
| Dataset | `luisroque/instruct-python-500k` (HuggingFace Datasets) |
| Data handling | pandas, NumPy |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A **Google AI Studio API key** ([get one here](https://aistudio.google.com/app/apikey))
- A GPU runtime is recommended for embedding 100k documents (the notebook was developed on a Google Colab T4).

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install -qU langchain langchain-community langchain-huggingface \
    langchain-google-genai chromadb rank_bm25 datasets
```

### Run

1. Open the notebook in **Google Colab** or Jupyter.
2. Run all cells in order. The first run downloads the dataset and embedding model, then builds the Chroma index. This is the slowest step, and the index is persisted to `./chroma_langchain`.
3. When prompted, enter your Google API key (it is read with `getpass`, so it is never stored in the notebook). Alternatively, set it beforehand:

   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```
4. Chat with the assistant in the final cell. Type `exit` to quit.

---

## Example

```
Python_RAG_Assistant(PyRexa)
Type 'exit' to end the conversation.

You: How do I reverse a list of tuples in Python without using reversed()?

Assistant:
1. Slicing:        tuples_list[::-1]     # new list
2. In place:       tuples_list.reverse() # returns None
3. Comprehension:  [tuples_list[i] for i in range(len(tuples_list) - 1, -1, -1)]
4. Recursion:      (see full answer in the notebook)
```

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| Sample size | `df_full.sample(n=...)` | `100000` |
| Embedding model | `HuggingFaceEmbeddings(model_name=...)` | `all-MiniLM-L6-v2` |
| Retrieval depth per retriever | `search_kwargs={"k": ...}` / `bm25_retriever.k` | `10` |
| Retriever weights | `EnsembleRetriever(weights=...)` | `[0.5, 0.5]` |
| Memory window | `Last5ChatMessageHistory(max_messages=...)` | `10` messages |
| LLM | `ChatGoogleGenerativeAI(model=..., temperature=...)` | `gemini-3.5-flash`, `0.2` |

---

## Project Structure

```
.
├── mid_project_orange.ipynb   # Full pipeline: data → index → RAG chain → chat loop
├── chroma_langchain/          # Persisted vector store (generated, git-ignored)
└── README.md
```

---

## Possible Improvements

- Add a **re-ranker** (e.g. a cross-encoder) on top of the hybrid results.
- Chunk long answers instead of storing them whole in metadata.
- Wrap the chain in a **Streamlit / Gradio** UI or a FastAPI service.
- Add a retrieval **evaluation** set (hit rate / MRR) to tune the BM25 vs. dense weights.
- Use the answer `score` metadata to favor higher-voted answers.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
> The dataset is subject to its own license. Please check its [Hugging Face page](https://huggingface.co/datasets/luisroque/instruct-python-500k).

---

## Author

**Ahmed Essam**: [GitHub](https://github.com/Ahmed-Essam01) · [LinkedIn](https://linkedin.com/in/ahmed-essam-ai

)
