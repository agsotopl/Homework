import streamlit as st

from openai import OpenAI
try:
    from anthropic import Anthropic
    anthropic_available = True
except Exception:
    Anthropic = None
    anthropic_available = False
import requests
from bs4 import BeautifulSoup
import glob
import os
import sys
import sqlite3

# verify sqlite3 version is >= 3.35.0
def _ensure_sqlite_version(min_version=(3, 35, 0)):
    try:
        ver = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
        if ver >= min_version:
            return True
    except Exception:
        pass

    try:
        
        from pysqlite3 import dbapi2 as pysqlite3_dbapi
        sys.modules["sqlite3"] = pysqlite3_dbapi
        return True
    except Exception:
        return False


if not _ensure_sqlite_version():
    st.error("Your system sqlite3 is too old for chromadb. Please install `pysqlite3-binary` in the environment and reload.`")
    st.stop()

import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader


st.title("HW 5: ChromaDB Optimizations")

# Chatbot Description
st.write("""
This chatbot uses advanced LLM models (OpenAI GPT-5 or Claude Opus 4.5) to have intelligent conversations.

**Conversation Memory:** The chatbot maintains a buffer of the last 5 interactions (10 messages, i.e. 5 user-assistant exchanges) 
along with a persistent system prompt. The system prompt includes any URL context you provide and is 
**never discarded** throughout the entire conversation, ensuring consistent context. This approach 
efficiently manages token usage while maintaining conversation coherence.
""")

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def build_system_prompt(url_1, url_2):
    """Build system prompt with URL context"""
    base_system_prompt = "You are a helpful assistant who explains things in a way that a 10-year-old can understand. Use simple words, short sentences, and lighthearted examples. After answering a question, always ask 'Do you want more info?' If the user says yes, provide more details and ask again. If the user says no, go back to asking 'How can I help?'"
    

    url_context = ""
    if url_1 or url_2:
        urls_to_fetch = [url_1, url_2]
        for i, url in enumerate(urls_to_fetch, 1):
            if url:
                content = read_url_content(url)
                if content:
                    url_context += f"\n\n--- Content from URL {i} ---\n{content[:2000]}"
    
    if url_context:
        return f"{base_system_prompt}\n\nYou have been provided with the following URL content as context:{url_context}"
    else:
        return base_system_prompt

openai_api_key = st.secrets.get("API_KEY")  
claude_api_key = st.secrets.get("CLAUDE_API_KEY")

if not openai_api_key:
    st.error("Missing API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()

openai_client = OpenAI(api_key=openai_api_key)

# Initialize Anthropic client
anthropic_client = None
if claude_api_key and anthropic_available:
    try:
        anthropic_client = Anthropic(api_key=claude_api_key)
    except Exception:
        anthropic_client = None
elif claude_api_key and not anthropic_available:
    st.sidebar.warning("Anthropic SDK not installed; Claude option will be disabled.")
elif not claude_api_key and anthropic_available:
    st.sidebar.warning("CLAUDE_API_KEY missing from secrets; Claude option will be disabled.")


def extract_text_from_pdf(path):
    """Extract text from a PDF file using PyPDF2."""
    try:
        reader = PdfReader(path)
        text_pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
        return "\n\n".join(text_pages)
    except Exception as e:
        return ""


def clean_text(text: str) -> str:
    """Clean extracted text for better snippet display.

    - Remove hyphenated line breaks (e.g. 'exam-\nple' -> 'example')
    - Replace newlines with spaces and collapse multiple spaces
    """
    if not text:
        return ""
    
    text = text.replace("-\n", "")
     
    text = text.replace("\n", " ")
    
    return " ".join(text.split())


def build_chroma_collection(openai_client, collection_name="Lab4Collection", embedding_model="text-embedding-3-small"):
    """
    Build or return a Chroma collection stored in st.session_state.Lab4_VectorDB.

    - Searches the workspace for PDF files (**/*.pdf)
    - Extracts text, uses filename as the document ID/key
    - Creates embeddings via OpenAI and stores documents in a ChromaDB collection
    - Stores the collection object in `st.session_state.Lab4_VectorDB`
    """
    if "Lab4_VectorDB" in st.session_state:
        return st.session_state.Lab4_VectorDB

    # Use html/directory
    workspace_root = os.getcwd()
    html_dir = os.path.join(workspace_root, "html")
    html_pattern = os.path.join(html_dir, "*.html")
    html_paths = [p for p in glob.glob(html_pattern) if ".venv" not in p and ".git" not in p]
    # Show HTML files in sidebar
    if html_paths:
        st.sidebar.write(f"Found {len(html_paths)} HTML file(s):")
        for p in html_paths:
            st.sidebar.write(f"- {os.path.relpath(p, workspace_root)}")
    if not html_paths:
        st.warning("No HTML files found in the workspace/html folder to build the Chroma collection.")
        st.session_state.Lab4_VectorDB = None
        return None

    # Use a persist directory so the DB is only created once and can be re-used across runs
    persist_directory = os.path.join(workspace_root, "chroma_db")
    chroma_client = chromadb.Client(Settings(persist_directory=persist_directory))

    
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        try:
            test_q = collection.count()
            if test_q and test_q > 0:
                st.sidebar.info(f"Loaded existing collection '{collection_name}' with {test_q} items from persist directory")
                st.session_state.Lab4_VectorDB = collection
                return collection
        except Exception:
            pass
    except Exception:
        collection = None

    # Create collection if it doesn't exist
    if collection is None:
        collection = chroma_client.create_collection(name=collection_name)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    # Chunking helper
    def two_part_chunk(text):
        if not text:
            return []
        length = len(text)
        if length < 500:
            # short doc, keep as single chunk
            return [text]
        mid = length // 2
        # try to find period near midpoint
        left = text.rfind('.', 0, mid)
        right = text.find('.', mid)
        split_at = None
        if left != -1 and mid - left < 200:
            split_at = left + 1
        elif right != -1 and right - mid < 200:
            split_at = right + 1
        else:
            # fallback to whitespace
            ws_left = text.rfind(' ', 0, mid)
            ws_right = text.find(' ', mid)
            if ws_left != -1 and mid - ws_left < 200:
                split_at = ws_left + 1
            elif ws_right != -1:
                split_at = ws_right + 1
            else:
                split_at = mid
        part1 = text[:split_at].strip()
        part2 = text[split_at:].strip()
        parts = [p for p in (part1, part2) if p]
        return parts

    for html_path in html_paths:
        try:
            with open(html_path, 'rb') as fh:
                soup = BeautifulSoup(fh, 'html.parser')
                raw_text = soup.get_text(separator=' ')
        except Exception:
            st.sidebar.warning(f"Failed to read/parse {os.path.basename(html_path)}")
            continue
        # Clean text
        text = clean_text(raw_text)
        if not text:
            st.sidebar.warning(f"No text extracted from {os.path.basename(html_path)}")
            continue
        base_id = os.path.basename(html_path)
        parts = two_part_chunk(text)
        for i, part in enumerate(parts):
            doc_id = f"{base_id}::part_{i}"
            ids.append(doc_id)
            documents.append(part)
            metadatas.append({"source": html_path, "filename": base_id, "part": i})

    # Batch embeddings via OpenAI client 
    for doc in documents:
        try:
            emb_resp = openai_client.embeddings.create(model=embedding_model, input=doc)
            emb = emb_resp.data[0].embedding
        except Exception as e:
            emb = None
        embeddings.append(emb)

    # Filter out docs where embedding failed
    final_ids, final_docs, final_meta, final_embs = [], [], [], []
    for i, emb in enumerate(embeddings):
        if emb is None:
            continue
        final_ids.append(ids[i])
        final_docs.append(documents[i])
        final_meta.append(metadatas[i])
        final_embs.append(emb)

    if final_ids:
        collection.add(ids=final_ids, documents=final_docs, metadatas=final_meta, embeddings=final_embs)
        try:
            chroma_client.persist()
        except Exception:
            pass
        st.sidebar.success(f"Added {len(final_ids)} document chunks to '{collection_name}' and persisted DB")
    else:
        st.sidebar.warning("No document chunks were added to the collection (embeddings may have failed).")

    st.session_state.Lab4_VectorDB = collection
    return collection


def relevant_course_info(query, collection, openai_client, embedding_model="text-embedding-3-small", top_k=10, max_files=5):
    """Search the ChromaDB collection for the input `query` and return
    a list of text snippets and filenames that are most relevant.

    This runs the vector search locally (not via model function-calling) and
    returns content you can inject into the LLM's system prompt.
    Returns: (retrieved_blocks, retrieved_filenames)
    """
    if not collection or not query:
        return [], []

    try:
        emb_resp = openai_client.embeddings.create(model=embedding_model, input=query)
        query_emb = emb_resp.data[0].embedding
        res = collection.query(query_embeddings=[query_emb], n_results=top_k, include=["metadatas", "documents", "distances"])
    except Exception:
        return [], []

    metas_list = res.get("metadatas", [])
    docs_list = res.get("documents", [])
    dists_list = res.get("distances", [])

    if metas_list and isinstance(metas_list[0], list):
        metas_for_query = metas_list[0]
        docs_for_query = docs_list[0] if docs_list else []
        dists_for_query = dists_list[0] if dists_list else []
    else:
        metas_for_query = metas_list
        docs_for_query = docs_list
        dists_for_query = dists_list

    best_per_file = {}
    for i, meta in enumerate(metas_for_query):
        try:
            filename = None
            if isinstance(meta, dict):
                filename = meta.get("filename") or os.path.basename(meta.get("source", ""))
            if not filename:
                filename = f"result_{i+1}"
            dist = dists_for_query[i] if i < len(dists_for_query) else None
            snippet = docs_for_query[i] if i < len(docs_for_query) else ""
            if filename not in best_per_file or (dist is not None and (best_per_file[filename][0] is None or dist < best_per_file[filename][0])):
                best_per_file[filename] = (dist, clean_text(snippet))
        except Exception:
            continue

    sorted_files = sorted(best_per_file.items(), key=lambda kv: (kv[1][0] if kv[1][0] is not None else float('inf')))

    retrieved_blocks = []
    retrieved_filenames = []
    for fname, (dist, snippet) in sorted_files[:max_files]:
        retrieved_filenames.append(fname)
        retrieved_blocks.append(f"Filename: {fname}\nSnippet: {snippet[:2000]}")

    return retrieved_blocks, retrieved_filenames


# Build collection once and store in session state
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Building ChromaDB collection from HTML files and persisting to disk..."):
        build_chroma_collection(openai_client)

st.sidebar.header("Model Settings")

model_provider = st.sidebar.radio(
    "Select AI Model:",
    ("OpenAI (GPT-5)", "Claude (Opus 4.5)"),
    index=0,
)

if model_provider == "OpenAI (GPT-5)":
    model = "gpt-5"
else:
    model = "claude-opus-4-5-20251101"

st.sidebar.header("URL Context (Optional)")
url_1 = st.sidebar.text_input("URL 1:", placeholder="https://example.com")
url_2 = st.sidebar.text_input("URL 2:", placeholder="https://example.com")




if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

if 'anthropic_client' not in st.session_state:
    st.session_state.anthropic_client = anthropic_client if anthropic_client is not None else None

# Track URL's for prompt context
current_urls = (url_1, url_2)

if 'messages' not in st.session_state:
    system_prompt = build_system_prompt(url_1, url_2)
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "assistant",
            "content": "Hey, how can I help you today?"
        }
    ]
    st.session_state.last_urls = current_urls
elif 'last_urls' in st.session_state and st.session_state.last_urls != current_urls:
    system_prompt = build_system_prompt(url_1, url_2)
    st.session_state.messages[0] = {
        "role": "system",
        "content": system_prompt
    }
    st.session_state.last_urls = current_urls

def maintain_buffer(messages, max_non_system_messages=10):
    """Keep system message (never discarded) and last 10 non-system messages (5 user-assistant exchanges)

    We interpret "5 interactions" as 5 user-assistant exchanges; each exchange has two messages,
    so we keep up to 10 non-system messages. The system message is always preserved.
    """
    system_message = None
    non_system_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg
        else:
            non_system_messages.append(msg)
    
    if len(non_system_messages) <= max_non_system_messages:
        return messages
    
    # Keep only last 6 messages
    filtered = non_system_messages[-max_non_system_messages:]
    

    if system_message:
        return [system_message] + filtered
    return filtered

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    chat_role = st.chat_message(msg["role"])
    chat_role.write(msg["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        
        if model_provider == "OpenAI (GPT-5)":
            chat_model = "gpt-5-mini"

        
            collection = st.session_state.get("Lab4_VectorDB")
            # Use the helper to perform vector search and assemble retrieved snippets
            retrieved_blocks, retrieved_filenames = relevant_course_info(
                prompt,
                collection,
                st.session_state.openai_client,
                embedding_model="text-embedding-3-small",
                top_k=10,
                max_files=5,
            )

            augmented_messages = list(st.session_state.messages)
            if retrieved_blocks:
                retrieved_text = "\n\n".join(retrieved_blocks)
                # Inject retrieved content directly into the system prompt for this call
                orig_system = augmented_messages[0]["content"] if augmented_messages and augmented_messages[0].get("role") == "system" else ""
                injected = (
                    orig_system
                    + "\n\n--- Retrieved documents (use these as context when helpful) ---\n"
                    + retrieved_text
                    + "\n---\nIf you use information from these retrieved documents, include a final line that starts with 'SOURCES:' and list the filenames you used."
                )
                augmented_messages[0] = {"role": "system", "content": injected}

            stream = st.session_state.openai_client.chat.completions.create(
                model=chat_model,
                messages=augmented_messages,
                stream=True,
            )
            response = st.write_stream(stream)
        else:
            system_message = None
            conversation_messages = []

            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append(msg)

            response_obj = st.session_state.anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_message,
                messages=conversation_messages,
            )
            response = response_obj.content[0].text
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Last 6 messages and system prompt
    st.session_state.messages = maintain_buffer(st.session_state.messages)
    
    # Latest Response
    st.rerun()