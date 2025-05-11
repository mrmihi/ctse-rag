"""
Streamlit RAG chatbot for a Milvus-stored PDF collection.
Run with:  streamlit run chatbot.py
"""
import os
import pathlib
import streamlit as st

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Milvus
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

GOOGLE_API_KEY = "AIzaSyBW6XKhpcRVHrthjvUBmyLHxTW7DJooWaA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY env var not set!"); st.stop()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION  = os.getenv("MILVUS_COLLECTION", "pdf_documents")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Milvus(
    connection_args={ "host": MILVUS_HOST, "port": MILVUS_PORT },
    collection_name=COLLECTION,
    embedding_function=embeddings,
    search_params={ "metric_type": "COSINE", "params": { "nprobe": 10 }},
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant. Use the context (with page numbers) "
        "to answer.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\n"
        "Answer:\n1. Summary – one sentence.\n2. Key Points – bullet list with page citations."
    ),
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")
st.caption("Ask anything about your PDF corpus; answers cite page numbers.")

if "msgs" not in st.session_state:
    st.session_state.msgs = []        # [(role, text)]

for role, text in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(text)

if q := st.chat_input("Ask your question…"):
    # show user msg
    with st.chat_message("user"):
        st.markdown(q)
    st.session_state.msgs.append(("user", q))

    # assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            res   = qa_chain({ "query": q })
            ans   = res["result"]
            st.markdown(ans)

            with st.expander("Show sources"):
                for d in res["source_documents"]:
                    src = pathlib.Path(d.metadata.get("source", "")).name
                    pg  = d.metadata.get("page", "?")
                    st.markdown(f"- **{src}**, page {pg}")

    st.session_state.msgs.append(("assistant", ans))

