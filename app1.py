from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv



# Existing imports and dotenv setup remain the same

# Updated Function for Vector Store Retrieval
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, allow_dangerous_deserialization=True)
    vector_store.save_local("faiss_index")
    return vector_store

# Define a Basic QA Chain
def get_retrieval_qa_chain(vector_store):
    """
    A retrieval-based QA chain for direct question answering.
    """
    retriever = vector_store.as_retriever()
    model = GooglePalm(model="gemini-pro", temperature=0.3)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Define a Conversational Retrieval Chain
def get_conversational_retrieval_chain(vector_store):
    """
    A conversational retrieval chain for multi-turn interactions with memory.
    """
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = GooglePalm(model="gemini-pro", temperature=0.3)

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return conversational_chain

# Multi-Prompt Chain for Enhanced Context Handling
def get_multi_prompt_chain():
    """
    Multi-prompt chain to select the best prompt for the context.
    """
    prompts = {
        "summary": PromptTemplate(template="Summarize this context:\n\n{context}", input_variables=["context"]),
        "qa": PromptTemplate(template="Answer the question based on context:\n\nContext:\n{context}\nQuestion:\n{question}", input_variables=["context", "question"]),
        "critical_thinking": PromptTemplate(template="Analyze and provide critical insights on:\n\n{context}", input_variables=["context"]),
    }

    model = GooglePalm(model="gemini-pro", temperature=0.3)

    # A dictionary maps intents or actions to prompts
    multi_prompt_chain = {
        "summary_chain": prompts["summary"],
        "qa_chain": prompts["qa"],
        "critical_chain": prompts["critical_thinking"],
    }
    return multi_prompt_chain

# Integrate into the Main Function
def user_input(user_question, chain_type="retrieval_qa"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings)

    if chain_type == "retrieval_qa":
        chain = get_retrieval_qa_chain(vector_store)
        response = chain.run(user_question)
    elif chain_type == "conversational":
        chain = get_conversational_retrieval_chain(vector_store)
        response = chain({"question": user_question})
    else:
        multi_chain = get_multi_prompt_chain()
        # Example of using the QA prompt for the multi-prompt chain
        qa_prompt = multi_chain["qa_chain"]
        response = qa_prompt.format(context=vector_store.similarity_search(user_question), question=user_question)

    print(response)
    st.write("Reply: ", response)

# Main Streamlit App
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    chain_option = st.selectbox(
        "Choose Chain Type", ["retrieval_qa", "conversational", "multi_prompt"]
    )

    if user_question:
        user_input(user_question, chain_type=chain_option)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
