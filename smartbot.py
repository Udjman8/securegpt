import streamlit as st
import os
import dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pinecone
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import random

# Load environment variables
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Gemini and Pinecone
genai.configure(api_key=GEMINI_API_KEY)
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index_name = "chatbot"

# Check if index exists
try:
    index = pc.Index(index_name)
except pinecone.exceptions.NotFoundException:
    st.error("Pinecone index not found. Please create the index before proceeding.")
    index = None

# Session state management
if "history" not in st.session_state:
    st.session_state.history = []
if "current_source" not in st.session_state:
    st.session_state.current_source = None

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def fetch_content(url, extra_links=5):
    """Fetches main webpage content and dynamically scrapes `extra_links` more pages."""
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        main_page_source = driver.page_source

        scraped_links = [url]
        extracted_texts = [clean_data(main_page_source)]

        extra_urls = driver.find_elements(By.TAG_NAME, "a")
        random.shuffle(extra_urls)  # Randomize links for better variety

        links_to_scrape = set()
        for link in extra_urls:
            try:
                href = link.get_attribute("href")
                if href and href.startswith("http") and href not in scraped_links:
                    links_to_scrape.add(href)
                    if len(links_to_scrape) >= extra_links:
                        break
            except Exception:
                continue  # Ignore stale elements

        st.subheader("Scraping the following URLs:")
        for idx, link in enumerate(links_to_scrape, start=1):
            st.write(f" {idx}. {link}")

        # Scrape additional links
        for link in links_to_scrape:
            try:
                driver.get(link)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                extracted_texts.append(clean_data(driver.page_source))
                scraped_links.append(link)
            except Exception as e:
                st.error(f"Error fetching {link}: {e}")

        driver.quit()

        # Store all extracted data in Pinecone
        for i, (text, link) in enumerate(zip(extracted_texts, scraped_links)):
            store_embeddings(text, link)  # Store embeddings with actual URL as doc_id

        return extracted_texts, scraped_links
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return [], []


def clean_data(html_content):
    """Removes scripts/styles and extracts clean text from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(['script', 'style']):
        script.decompose()
    return soup.get_text(separator=' ', strip=True)

def store_embeddings(text, doc_id):
    """Stores text embeddings in Pinecone."""
    try:
        words = text.split()
        chunks = [" ".join(words[i:i+1000]) for i in range(0, len(words), 1000)]
        embeddings = embed_model.encode(chunks).tolist()
        vectors = [(f"{doc_id}_{i}", embeddings[i], {"text": chunks[i]}) for i in range(len(chunks))]
        if index:
            index.upsert(vectors)
    except Exception as e:
        st.error(f"Error storing embeddings in Pinecone: {e}")

def retrieve_relevant_text(query):
    """Fetches relevant text from Pinecone based on query."""
    try:
        query_embedding = embed_model.encode([query]).tolist()[0]
        if index:
            results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
            if results['matches']:
                return "\n".join([match['metadata']['text'] for match in results['matches']])
        return "No relevant text found."
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return "Error querying Pinecone."

def send_to_gemini(query, retrieved_text, source):
    """Generates AI response using Gemini."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Given the following query and relevant extracted text, generate a concise response.
        
        Query: {query}
        Extracted Text:
        {retrieved_text}
        
        Mention the source name as source: {source}.
        """
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        st.error(f"Error generating content with Gemini: {e}")
        return f"Error generating content with Gemini: {e}"

st.set_page_config(page_title="SECUREGPT", layout="wide")

st.markdown("""
    <style>
        .stChatMessage { padding: 10px; border-radius: 10px; margin: 5px 0; }
        .user { background-color: #dcf8c6; align-self: flex-end; }
        .bot { background-color: #f1f0f0; align-self: flex-start; }
        .stTextInput>div>div>input { border-radius: 20px; padding: 10px; }
        .stButton>button { border-radius: 20px; padding: 8px 16px; }
    </style>
""", unsafe_allow_html=True)

st.title(" SECUREGPT")

# Sidebar with chat history
# with st.sidebar:
#     st.header(" Source History")
#     for entry in st.session_state.history:
#         st.write(entry)
#     if st.button("Clear History"):
#         st.session_state.history = []  # Reset session history
#         st.session_state.current_source = None  # Reset source selection
#         st.session_state.query = ""  # Reset input field
#         st.session_state.pop("uploaded_file", None)
#         st.session_state.pop("url", None)
#         if index:
#             stats = index.describe_index_stats()
#             if stats.get("total_vector_count", 0) > 0:  # Only delete if data exists
#                 index.delete(delete_all=True)
#         st.success("History cleared! 🔄")
#         time.sleep(2)
#         st.rerun()

# Sidebar with chat history
# Sidebar with chat history
with st.sidebar:
    st.header("Source History")
    for idx, entry in enumerate(st.session_state.history):
        question = entry.split("Q: ")[1].split("\nA: ")[0]
        if st.button(f"{idx+1}. {question}", key=f"history_{idx}"):
            st.session_state.query = question
            st.session_state.selected_response = entry.split("\nA: ")[1]  # Store response
            st.session_state.clear_main = True  # Indicate that the main page should clear
            break  # Prevent multiple reruns

    if st.button("Clear History"):
        st.session_state.history = []  # Reset session history
        st.session_state.current_source = None  # Reset source selection
        st.session_state.query = ""  # Reset input field
        st.session_state.pop("uploaded_file", None)
        st.session_state.pop("url", None)
        st.session_state.pop("selected_response", None)  # Clear selected response
        st.session_state.pop("clear_main", None)  # Reset main page flag
        if index:
            stats = index.describe_index_stats()
            if stats.get("total_vector_count", 0) > 0:  # Only delete if data exists
                index.delete(delete_all=True)
        st.success("History cleared! 🔄")
        time.sleep(1.5)
        st.rerun()


# Input selection (PDF or Web Scraping)
input_type = st.radio("Choose Input Type:", ("📄 Add PDF", "🌍 Add Link"))

if input_type == "📄 Add PDF":
    uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])
    if uploaded_file:
        if st.session_state.current_source != uploaded_file.name:
            st.session_state.current_source = uploaded_file.name
            st.session_state.history = []
            if index:
                stats = index.describe_index_stats()
                if stats.get("total_vector_count", 0) > 0:  # Only delete if data exists
                   index.delete(delete_all=True)
        with st.spinner(" Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        if extracted_text:
            with st.spinner(" Storing embeddings..."):
                store_embeddings(extracted_text, uploaded_file.name)
            st.success(" PDF processed successfully!")

elif input_type == "🌍 Add Link":
    url = st.text_input(" Enter a website URL")
    extra_links = st.slider(" Number of extra links to scrape:", 1, 10, 3)  # Slider instead of selectbox
    if st.button(" Scrape"):
        if st.session_state.current_source != url:
            st.session_state.current_source = url
            st.session_state.history = []
            if index:
               stats = index.describe_index_stats()
               if stats.get("total_vector_count", 0) > 0:  # Only delete if data exists
                  index.delete(delete_all=True)
        with st.spinner(" Fetching webpage content..."):
            extracted_texts, scraped_links = fetch_content(url, extra_links)
        
        if extracted_texts:
            with st.spinner(" Storing embeddings..."):
                for i, text in enumerate(extracted_texts):
                    store_embeddings(text, f"{url}_page_{i+1}")
            st.success(f" Scraped {len(scraped_links)} links successfully!")

# Chat UI
# st.markdown("---")
# st.subheader(" Chat with SECUREGPT")
# query = st.text_input(" Ask me anything:", key="query")

# if st.button(" Submit Query"):
#     if query:
#         retrieved_text = retrieve_relevant_text(query)
#         response = send_to_gemini(query, retrieved_text, st.session_state.current_source)
#         st.session_state.history.append(f"Q: {query}\nA: {response}")

#         # Display chat messages like ChatGPT
#         with st.chat_message("user"):
#             st.write(f"👤 **You:** {query}")
#         with st.chat_message("bot"):
#             st.write(f" **SECUREGPT:** {response}")
st.markdown("---")
st.subheader("Chat with SECUREGPT")

# If a history item was clicked, show it
if "selected_response" in st.session_state and st.session_state.selected_response:
    with st.chat_message("user"):
        st.write(f"👤 **You:** {st.session_state.query}")
    with st.chat_message("bot"):
        st.write(f"🤖 **SECUREGPT:** {st.session_state.selected_response}")
    st.session_state.pop("selected_response")  # Remove after showing

# Main chat input
query = st.text_input("Ask me anything:", key="query")

if st.button("Submit Query"):
    if query:
        retrieved_text = retrieve_relevant_text(query)
        response = send_to_gemini(query, retrieved_text, st.session_state.current_source)
        st.session_state.history.append(f"Q: {query}\nA: {response}")

        # Show only the latest query in the main chat
        with st.chat_message("user"):
            st.write(f"👤 **You:** {query}")
        with st.chat_message("bot"):
            st.write(f"🤖 **SECUREGPT:** {response}")

        # Clear history-related state
        st.session_state.pop("selected_response", None)
        st.session_state.pop("clear_main", None)
