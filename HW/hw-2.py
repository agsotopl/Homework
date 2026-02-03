import requests
import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
from anthropic import Anthropic


# Load API key's 
openai_api_key = st.secrets.get("API_KEY")  
claude_api_key = st.secrets.get("CLAUDE_API_KEY")  

if not openai_api_key:
    st.error("Missing API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()

if not claude_api_key:
    st.error("Missing CLAUDE_API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()

# Prepare clients
openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = Anthropic(api_key=claude_api_key)

client = OpenAI(api_key=openai_api_key)

# Header 
st.title("HW 2")
st.write("Enter the URL and select a summary style in the sidebar.")

# Summary Selection
st.sidebar.header("Summary Style")

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

summary_choice = st.sidebar.radio(
    "Choose one:",
    (
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points",
    ),
    index=0,
)

# Language selection 
st.sidebar.header("Language Settings")

output_language = st.sidebar.selectbox(
    "Select output language:",
    ("English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean")
)

# Model Selection 
st.sidebar.header("Model Settings")

model_provider = st.sidebar.radio(
    "Select AI Model:",
    ("GPT-5-Nano", "Claude Opus 4.5"),
    index=0,
)

if model_provider == "GPT-5-Nano":
    model = "gpt-5-nano"
else:  
    model = "claude-opus-4-5-20251101"

# Summary Selection Logic
if summary_choice == "100 words":
    format_instruction = (
        "Summarize the document in exactly 100 words. "
        "No heading or bullet points, only plain text."
    )
elif summary_choice == "2 connecting paragraphs":
    format_instruction = (
        "Summarize the document in exactly two connected paragraphs. "
        "No bullet points or headings."
    )
else:
    format_instruction = (
        "Summarize the document in exactly 5 bullet points. "
        "Each bullet must be a complete sentence."
    )

# URL input
url_input = st.text_input("Enter the web address URL:")

generate = st.button("Generate Summary", disabled=not url_input)

# Generate Summary
if url_input and generate:
    st.spinner("Reading content from URL...")
    content = read_url_content(url_input)
    
    if content is None:
        st.error(f"Failed to retrieve content from {url_input}")
    elif not content.strip():
        st.error("The URL returned no readable content.")
    else:
        st.spinner("Generating summary")
        try:
            if model_provider == "GPT-5-Nano":
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that summarizes web content.",
                        },
                        {
                            "role": "user",
                            "content": f"{format_instruction}\n\nContent to summarize:\n\n{content}\n\nPlease provide your response in {output_language}.",
                        },
                    ],
                )
                summary = response.choices[0].message.content
            else:  
                response = anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"You are a helpful assistant that summarizes web content.\n\n{format_instruction}\n\nContent to summarize:\n\n{content}\n\nPlease provide your response in {output_language}.",
                    },
                ],
            )
                summary = response.content[0].text
            
            st.write(summary)
        except Exception as e:
            st.error(f"Error generating summary: {e}")

