# RAG Question Answering over Arabic Wikipedia

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about a topic using its **Arabic Wikipedia** article, with a Groq-hosted LLM generating the answers from retrieved context.

## What it does
1. Loads an Arabic Wikipedia article with `WikipediaLoader` (`lang="ar"`).
2. Splits the text into overlapping chunks using NLTK (500 chars, 50 overlap).
3. Embeds the chunks with a HuggingFace embedding model and stores them in a **Chroma** vector store.
4. Retrieves the most relevant chunks for a question and passes them to the LLM (via **Groq**).
5. Returns structured answers parsed with a Pydantic output parser.

## Stack
`LangChain` · `Groq` · `HuggingFace embeddings` · `ChromaDB` · `NLTK` · `Pydantic`

## Run it
Open `QNA.ipynb` in Google Colab (the notebook reads secrets from Colab's *Secrets* tab).

Add this secret in Colab:
- `GROQ_API_KEY`: get one from [console.groq.com](https://console.groq.com)

Or locally:
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
```
> The notebook uses `google.colab.userdata`; when running locally replace it with `os.environ["GROQ_API_KEY"]`.
