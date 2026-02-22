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
import json

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
    st.error("Your system sqlite3 is too old for chromadb. Please install `pysqlite3-binary` in the environment and reload.")
    st.stop()

import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader


st.title("HW5: ChromaDB Chatbot with Function Calling")

st.write("""
This chatbot uses advanced LLM models (OpenAI GPT-4o or Claude Opus 4.5) to answer questions
about your documents using **Retrieval-Augmented Generation (RAG)**.

**How it works:** When you ask a question, the model uses a tool call to invoke `relevant_course_info`,
which performs a vector search over your ChromaDB collection. The retrieved document snippets are
then passed back to the model so it can formulate an informed answer.

**Conversation Memory:** The chatbot maintains a buffer of the last 5 interactions (10 messages)
along with a persistent system prompt that is **never discarded**.
""")


def read_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


def build_system_prompt(url_1, url_2):
    """Build system prompt with optional URL context."""
    base_system_prompt = (
        "You are a helpful assistant who explains things in a way that a 10-year-old can understand. "
        "Use simple words, short sentences, and lighthearted examples. "
        "After answering a question, always ask 'Do you want more info?' "
        "If the user says yes, provide more details and ask again. "
        "If the user says no, go back to asking 'How can I help?'"
    )

    url_context = ""
    for i, url in enumerate([url_1, url_2], 1):
        if url:
            content = read_url_content(url)
            if content:
                url_context += f"\n\n--- Content from URL {i} ---\n{content[:2000]}"

    if url_context:
        return f"{base_system_prompt}\n\nYou have been provided with the following URL content as context:{url_context}"
    return base_system_prompt


def clean_text(text: str) -> str:
    """Remove hyphenated line breaks, collapse whitespace."""
    if not text:
        return ""
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    return " ".join(text.split())


# ChromaDB builder

def build_chroma_collection(openai_client, collection_name="Lab4Collection",
                             embedding_model="text-embedding-3-small"):
    """Build or return a Chroma collection stored in st.session_state.Lab4_VectorDB."""
    if "Lab4_VectorDB" in st.session_state:
        return st.session_state.Lab4_VectorDB

    workspace_root = os.getcwd()
    html_dir = os.path.join(workspace_root, "html")
    html_paths = [
        p for p in glob.glob(os.path.join(html_dir, "*.html"))
        if ".venv" not in p and ".git" not in p
    ]

    if html_paths:
        st.sidebar.write(f"Found {len(html_paths)} HTML file(s):")
        for p in html_paths:
            st.sidebar.write(f"- {os.path.relpath(p, workspace_root)}")

    if not html_paths:
        st.warning("No HTML files found in the workspace/html folder to build the Chroma collection.")
        st.session_state.Lab4_VectorDB = None
        return None

    persist_directory = os.path.join(workspace_root, "chroma_db")
    chroma_client = chromadb.Client(Settings(persist_directory=persist_directory))

    try:
        collection = chroma_client.get_collection(name=collection_name)
        if collection.count() > 0:
            st.sidebar.info(
                f"Loaded existing collection '{collection_name}' with {collection.count()} items."
            )
            st.session_state.Lab4_VectorDB = collection
            return collection
    except Exception:
        collection = None

    if collection is None:
        collection = chroma_client.create_collection(name=collection_name)

    def two_part_chunk(text):
        if not text:
            return []
        length = len(text)
        if length < 500:
            return [text]
        mid = length // 2
        left = text.rfind('.', 0, mid)
        right = text.find('.', mid)
        split_at = None
        if left != -1 and mid - left < 200:
            split_at = left + 1
        elif right != -1 and right - mid < 200:
            split_at = right + 1
        else:
            ws_left = text.rfind(' ', 0, mid)
            ws_right = text.find(' ', mid)
            if ws_left != -1 and mid - ws_left < 200:
                split_at = ws_left + 1
            elif ws_right != -1:
                split_at = ws_right + 1
            else:
                split_at = mid
        return [p for p in (text[:split_at].strip(), text[split_at:].strip()) if p]

    ids, documents, metadatas, embeddings = [], [], [], []

    for html_path in html_paths:
        try:
            with open(html_path, 'rb') as fh:
                soup = BeautifulSoup(fh, 'html.parser')
                raw_text = soup.get_text(separator=' ')
        except Exception:
            st.sidebar.warning(f"Failed to read/parse {os.path.basename(html_path)}")
            continue

        text = clean_text(raw_text)
        if not text:
            st.sidebar.warning(f"No text extracted from {os.path.basename(html_path)}")
            continue

        base_id = os.path.basename(html_path)
        for i, part in enumerate(two_part_chunk(text)):
            ids.append(f"{base_id}::part_{i}")
            documents.append(part)
            metadatas.append({"source": html_path, "filename": base_id, "part": i})

    for doc in documents:
        try:
            emb_resp = openai_client.embeddings.create(model=embedding_model, input=doc)
            embeddings.append(emb_resp.data[0].embedding)
        except Exception:
            embeddings.append(None)

    final_ids, final_docs, final_meta, final_embs = [], [], [], []
    for i, emb in enumerate(embeddings):
        if emb is None:
            continue
        final_ids.append(ids[i])
        final_docs.append(documents[i])
        final_meta.append(metadatas[i])
        final_embs.append(emb)

    if final_ids:
        collection.add(
            ids=final_ids, documents=final_docs,
            metadatas=final_meta, embeddings=final_embs
        )
        try:
            chroma_client.persist()
        except Exception:
            pass
        st.sidebar.success(f"Added {len(final_ids)} chunks to '{collection_name}'.")
    else:
        st.sidebar.warning("No chunks were added (embeddings may have failed).")

    st.session_state.Lab4_VectorDB = collection
    return collection


# RAG retrieval function

def relevant_course_info(query: str, collection, openai_client,
                          embedding_model="text-embedding-3-small",
                          top_k=10, max_files=5):
    """
    Vector-search the ChromaDB collection for `query`.
    Returns (retrieved_blocks, retrieved_filenames).
    This is also exposed as an OpenAI tool so the model can call it directly.
    """
    if not collection or not query:
        return [], []

    try:
        emb_resp = openai_client.embeddings.create(model=embedding_model, input=query)
        query_emb = emb_resp.data[0].embedding
        res = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
    except Exception:
        return [], []

    metas_list = res.get("metadatas", [])
    docs_list = res.get("documents", [])
    dists_list = res.get("distances", [])

    metas_for_query = metas_list[0] if metas_list and isinstance(metas_list[0], list) else metas_list
    docs_for_query = docs_list[0] if docs_list and isinstance(docs_list[0], list) else docs_list
    dists_for_query = dists_list[0] if dists_list and isinstance(dists_list[0], list) else dists_list

    best_per_file = {}
    for i, meta in enumerate(metas_for_query):
        filename = None
        if isinstance(meta, dict):
            filename = meta.get("filename") or os.path.basename(meta.get("source", ""))
        if not filename:
            filename = f"result_{i+1}"
        dist = dists_for_query[i] if i < len(dists_for_query) else None
        snippet = docs_for_query[i] if i < len(docs_for_query) else ""
        if filename not in best_per_file or (
            dist is not None and (best_per_file[filename][0] is None or dist < best_per_file[filename][0])
        ):
            best_per_file[filename] = (dist, clean_text(snippet))

    sorted_files = sorted(best_per_file.items(),
                          key=lambda kv: (kv[1][0] if kv[1][0] is not None else float('inf')))

    retrieved_blocks, retrieved_filenames = [], []
    for fname, (_, snippet) in sorted_files[:max_files]:
        retrieved_filenames.append(fname)
        retrieved_blocks.append(f"Filename: {fname}\nSnippet: {snippet[:2000]}")

    return retrieved_blocks, retrieved_filenames


# OpenAI tool definition

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "relevant_course_info",
            "description": (
                "Search the course document collection for information relevant to the query. "
                "Use this whenever the user asks a question that may be answered by course materials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the course documents."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def handle_tool_call(tool_call, collection, openai_client):
    """Execute a tool call and return the result string."""
    args = json.loads(tool_call.function.arguments)
    query = args.get("query", "")
    retrieved_blocks, retrieved_filenames = relevant_course_info(
        query, collection, openai_client
    )
    if retrieved_blocks:
        result = "\n\n".join(retrieved_blocks)
        result += f"\n\nSOURCES: {', '.join(retrieved_filenames)}"
    else:
        result = "No relevant documents found for that query."
    return result


#OpenAI chat 

def openai_chat_with_tools(messages, openai_client, collection, model="gpt-4o"):
    """
    Run the OpenAI chat completion loop with tool calling.
    Returns the final assistant text response.
    """
    working_messages = list(messages)

    while True:
        response = openai_client.chat.completions.create(
            model=model,
            messages=working_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            working_messages.append(choice.message)

            
            for tool_call in choice.message.tool_calls:
                tool_result = handle_tool_call(tool_call, collection, openai_client)
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })
            continue

        
        return choice.message.content


# Claude RAG helper

def claude_chat_with_rag(messages, system_prompt, anthropic_client, collection,
                          openai_client, model="claude-opus-4-5-20251101"):
    """
    Run a Claude chat completion augmented with RAG.
    Retrieves relevant documents for the latest user message and injects
    them into the system prompt before calling the Claude API.
    """
    # Find the latest user message to use as the retrieval query
    query = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            query = msg["content"]
            break

    retrieved_blocks, retrieved_filenames = relevant_course_info(
        query, collection, openai_client
    )

    augmented_system = system_prompt
    if retrieved_blocks:
        retrieved_text = "\n\n".join(retrieved_blocks)
        augmented_system = (
            system_prompt
            + "\n\n--- Retrieved documents (use these as context when helpful) ---\n"
            + retrieved_text
            + f"\n\nSOURCES: {', '.join(retrieved_filenames)}"
            + "\n---\nIf you use information from these retrieved documents, cite the filenames at the end of your response."
        )

    response_obj = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        system=augmented_system,
        messages=messages,
    )
    return response_obj.content[0].text


# ── Buffer maintenance ─────────────────────────────────────────────────────────

def maintain_buffer(messages, max_non_system_messages=10):
    """
    Keep the system message (never discarded) and the last 10 non-system
    messages (5 user-assistant exchanges). Tool messages are included in the count.
    """
    system_message = None
    non_system_messages = []

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system_message = msg
        else:
            non_system_messages.append(msg)

    if len(non_system_messages) <= max_non_system_messages:
        return messages

    filtered = non_system_messages[-max_non_system_messages:]
    return ([system_message] + filtered) if system_message else filtered


# API keys & clients

openai_api_key = st.secrets.get("API_KEY")
claude_api_key = st.secrets.get("CLAUDE_API_KEY")

if not openai_api_key:
    st.error("Missing API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()

openai_client = OpenAI(api_key=openai_api_key)

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


# Build collection 

if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Building ChromaDB collection from HTML files…"):
        build_chroma_collection(openai_client)

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = openai_client

if 'anthropic_client' not in st.session_state:
    st.session_state.anthropic_client = anthropic_client


# Sidebar 

st.sidebar.header("Model Settings")

# Determine available providers
provider_options = ["OpenAI (GPT-4o)"]
if anthropic_client is not None:
    provider_options.append("Claude (Opus 4.5)")

model_provider = st.sidebar.radio("Select AI Model:", provider_options, index=0)

st.sidebar.header("URL Context (Optional)")
url_1 = st.sidebar.text_input("URL 1:", placeholder="https://example.com")
url_2 = st.sidebar.text_input("URL 2:", placeholder="https://example.com")

# Session State

current_urls = (url_1, url_2)

if 'messages' not in st.session_state:
    system_prompt = build_system_prompt(url_1, url_2)
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hey, how can I help you today?"},
    ]
    st.session_state.last_urls = current_urls
elif st.session_state.get('last_urls') != current_urls:
    system_prompt = build_system_prompt(url_1, url_2)
    st.session_state.messages[0] = {"role": "system", "content": system_prompt}
    st.session_state.last_urls = current_urls




for msg in st.session_state.messages:
    if not isinstance(msg, dict):
        continue
    role = msg.get("role")
    if role in ("system", "tool"):
        continue
    
    if role == "assistant" and not isinstance(msg.get("content"), str):
        continue
    with st.chat_message(role):
        st.markdown(msg["content"])


# Chat input & response 
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    collection = st.session_state.get("Lab4_VectorDB")

    with st.chat_message("assistant"):
        if model_provider == "OpenAI (GPT-4o)":
            # Build messages list 
            api_messages = [
                m for m in st.session_state.messages
                if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant", "tool")
            ]
            response = openai_chat_with_tools(
                api_messages,
                st.session_state.openai_client,
                collection,
                model="gpt-4o",
            )
            st.markdown(response)

        else:  # Claude
            system_message = ""
            conversation_messages = []
            for msg in st.session_state.messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "system":
                    system_message = msg["content"]
                elif msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
                    conversation_messages.append(msg)

            response = claude_chat_with_rag(
                conversation_messages,
                system_message,
                st.session_state.anthropic_client,
                collection,
                st.session_state.openai_client,
                model="claude-opus-4-5-20251101",
            )
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages = maintain_buffer(st.session_state.messages)
    st.rerun()