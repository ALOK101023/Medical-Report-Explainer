import streamlit as st
import os
import feedparser
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Setup ---
st.set_page_config(page_title="Medical Report Explainer", page_icon="🏥", layout="wide")

# --- RAG Logic for Medical Reports ---
@st.cache_resource
def process_medical_pdf(uploaded_file):
    # 1. Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    # 3. Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 4. LLM & Prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    prompt = PromptTemplate(
        template="""
        You are a helpful medical assistant. Explain medical reports in simple, bilingual language (English and Hindi).
        
        For each value or test result mentioned:
        - Explain what it means in simple terms.
        - Clearly state if it appears Normal or Abnormal based on the report ranges.
        - Suggest what abnormal values could mean and what to do next.
        
        Answer ONLY from the context. If context is missing, say you don't know.
        Always end with: "Please consult your doctor for proper medical advice."

        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. The Chain
    main_chain = (
        RunnableParallel({'context': retriever | RunnableLambda(format_docs), 'question': RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )
    return main_chain

# --- Sidebar: Health News & Disclaimer ---
with st.sidebar:
    st.title("🏥 Health Hub")
    st.warning("⚠️ For educational purposes only. Not a medical diagnosis.")
    st.markdown("---")
    
    st.markdown("### 🌐 Bilingual / द्विभाषी\nAsk in **English** or **हिंदी**")
    st.divider()
    
    st.markdown("### 🩺 Health News (Live)")
    try:
        # Fetching Health News via PIB or Medical RSS
        health_feed = feedparser.parse("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1")
        for entry in health_feed.entries[:4]:
            if "Health" in entry.title or "Medical" in entry.title:
                st.markdown(f"**• [{entry.title}]({entry.link})**")
    except:
        st.write("News feed refreshing...")
    
    if st.button("🗑 Clear Report & Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main App ---
st.title("🏥 Medical Report Explainer")
st.caption("Understand your lab results in simple language | अंग्रेजी और हिंदी सहायता")

# API Key Check
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Missing API Key in Streamlit Secrets!")
    st.stop()

# Upload Section
uploaded_file = st.file_uploader("Upload your medical report (PDF)", type="pdf")

if uploaded_file:
    st.success("✅ Report Loaded!")
    
    # --- Suggestion Pills ---
    st.write("### Quick Questions:")
    medical_suggestions = [
        "What are the abnormal values?",
        "Explain my Hemoglobin levels",
        "Is my blood sugar normal?",
        "Summary of this report"
    ]
    selected_pill = st.pills("Choose a quick question:", medical_suggestions, selection_mode="single", label_visibility="collapsed")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Combine Input
    user_query = None
    if selected_pill:
        user_query = selected_pill
    if chat_input := st.chat_input("Ask a specific question about your report..."):
        user_query = chat_input

    # Response Logic
    if user_query:
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing Medical Data..."):
                    chain = process_medical_pdf(uploaded_file)
                    response = chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👆 Please upload a PDF report to start the explanation.")
