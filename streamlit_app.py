import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
from langchain_community.vectorstores import Pinecone
import os
import torch


os.environ["PINECONE_API_KEY"] = ""  
os.environ["PINECONE_ENVIRONMENT"] = "" 

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_name = "mychatter"

def load_and_split_docs(directory, chunk_size=500, chunk_overlap=20):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def create_or_load_index(docs):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeddings.embed_query("test")), metric='cosine')
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

def get_similar_docs(query, index, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs


st.title("Document Q&A")

uploaded_files = st.file_uploader("Upload documents", type=['pdf','txt','docx'], accept_multiple_files=True)
if uploaded_files:
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    docs = load_and_split_docs(temp_dir)
    index = create_or_load_index(docs)

    query = st.text_input("Enter your question:")
    if query:
        similar_docs = get_similar_docs(query, index)
        if similar_docs:
            st.subheader("Relevant Documents:")
            for doc in similar_docs:
                st.write(doc.page_content)
        else:
            st.write("No relevant documents found.")
