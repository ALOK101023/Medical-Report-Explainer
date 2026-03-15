import streamlit as st
import os
import feedparser
import pytesseract
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Setup ---
st.set_page_config(page_title="Medical Report Explainer", page_icon="🏥", layout="wide")

@st.cache_resource
def process_medical_pdf(file_bytes):
    # 1. Try standard text extraction first
    from io import BytesIO
    pdf_file = BytesIO(file_bytes)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # 2. OCR Fallback (If the PDF is a scanned image)
    if not text.strip():
        with st.spinner("🔍 Scanned report detected. Performing OCR..."):
            images = convert_from_bytes(file_bytes)
            for img in images:
                text += pytesseract.image_to_string(img)

    if not text.strip():
        st.error("🚫 Could not read any text. Please ensure the image is clear.")
        st.stop()

    # 3. RAG Processing
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    prompt = PromptTemplate(
        template="""You are a medical assistant. Explain the report in simple English and Hindi.
        Explain if values are Normal/Abnormal and what to do next.
        
        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        RunnableParallel({'context': retriever | RunnableLambda(format_docs), 'question': RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )

# --- Sidebar ---
with st.sidebar:
    st.title("🏥 Health Hub")
    st.warning("⚠️ For education only. Consult a doctor.")
    st.markdown("### 🌐 Bilingual / द्विभाषी Support")
    st.divider()
    st.markdown("### 🔔 Health News")
    try:
        feed = feedparser.parse("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1")
        for entry in feed.entries[:3]:
            st.markdown(f"**• [{entry.title}]({entry.link})**")
    except:
        pass

# --- Main App ---
st.title("🏥 Medical Report Explainer")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Add OPENAI_API_KEY to Secrets!")
    st.stop()

uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type="pdf")

if uploaded_file:
    # We pass bytes to the function for OCR processing
    file_bytes = uploaded_file.read()
    st.success("✅ File received!")

    # Suggestions
    medical_suggestions = ["What is abnormal?", "Explain Hemoglobin", "Is sugar level okay?"]
    selected_pill = st.pills("Quick Questions:", medical_suggestions, selection_mode="single")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = None
    if selected_pill: user_query = selected_pill
    if chat_input := st.chat_input("Ask about your report..."): user_query = chat_input

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)
        
        with st.chat_message("assistant"):
            chain = process_medical_pdf(file_bytes)
            response = chain.invoke(user_query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
