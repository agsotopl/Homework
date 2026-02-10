import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup


st.title("HW 3: Chatbot")

# Chatbot Description
st.write("""
This chatbot uses advanced LLM models (OpenAI GPT-5 or Claude Opus 4.5) to have intelligent conversations.

**Conversation Memory:** The chatbot maintains a buffer of the last 6 messages (3 user-assistant exchanges) 
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
    
    # Fetch URL content
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

if not claude_api_key:
    st.error("Missing CLAUDE_API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()

openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = Anthropic(api_key=claude_api_key)

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
    st.session_state.anthropic_client = Anthropic(api_key=claude_api_key)

# Initialize messages with URL context or update if URLs changed
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
    # URLs have changed, update system prompt
    system_prompt = build_system_prompt(url_1, url_2)
    st.session_state.messages[0] = {
        "role": "system",
        "content": system_prompt
    }
    st.session_state.last_urls = current_urls

def maintain_buffer(messages, max_non_system_messages=6):
    """Keep system message (never discarded) and last 6 non-system messages (3 user-assistant exchanges)"""
    system_message = None
    non_system_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg
        else:
            non_system_messages.append(msg)
    
    if len(non_system_messages) <= max_non_system_messages:
        return messages
    
    # Keep only the last max_non_system_messages
    filtered = non_system_messages[-max_non_system_messages:]
    
    # Prepend system message (always kept)
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
            stream = st.session_state.openai_client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                stream=True,
            )
            response = st.write_stream(stream)
        else:
            # Extract system message and non-system messages for Claude
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
                system=system_message,  # Separate system parameter
                messages=conversation_messages,  # Only user/assistant messages
            )
            response = response_obj.content[0].text
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Maintain buffer - keep system message + last 6 non-system messages
    st.session_state.messages = maintain_buffer(st.session_state.messages)
    
    # Show latest response
    st.rerun()