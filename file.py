import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent
import fitz 
import docx2txt
import os

# --- Helpers ---
def read_file(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")
st.title("📄 PDF, DOCX, TXT Q&A Assistant")

uploaded_files = st.file_uploader("Upload multiple files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    # Read all files
    raw_text = ""
    for file in uploaded_files:
        raw_text += read_file(file) + "\n"

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    # Embeddings + FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # QA Chain
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Define tool for agent
    def answer_from_documents(query: str):
        docs = vectorstore.similarity_search(query, k=4)
        return chain.run(input_documents=docs, question=query)

    tools = [
        Tool(
            name="MultiFileQA",
            func=answer_from_documents,
            description="Use this to answer questions from the uploaded files."
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent_type="zero-shot-react-description",
        verbose=True
    )

    # Chat interface
    query = st.text_input("Ask a question about the uploaded documents")
    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            answer = agent.run(query)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("Please upload at least one .pdf, .docx, or .txt file.")
