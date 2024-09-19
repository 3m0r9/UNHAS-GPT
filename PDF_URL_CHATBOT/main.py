import os
import streamlit as st
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import requests
import io
import pandas as pd
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings
)
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from htmlTemplates import bot_template, css, user_template


def get_pdf_text(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_file = response.content
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return None


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    EMBEDDINGS_OPENAI = int(os.getenv("EMBEDDINGS_OPENAI"))
    if bool(EMBEDDINGS_OPENAI):
        st.write("-> OpenAI Embeddings")
        embeddings = OpenAIEmbeddings()
    else:
        st.write("-> HuggingFace Embeddings")
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Download and save the documents to Google Drive
    drive_service = build('drive', 'v3', credentials=credentials)

    for pdf_url in df['Document_url']:
        file_name = pdf_url.split('/')[-1]
        response = requests.get(pdf_url)
        if response.status_code == 200:
            file_data = response.content
            # Create a new file in Google Drive
            file_metadata = {'name': file_name}
            media = MediaIoBaseDownload(io.BytesIO(file_data))
            file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            st.write(f"Document '{file_name}' has been saved to Google Drive with ID: {file['id']}")
        else:
            st.write(f"Failed to download the document from URL: {pdf_url}")


def main():
    load_dotenv()
    credentials = service_account.Credentials.from_service_account_file('credentials.json')

    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = ""
                for pdf_doc in pdf_docs:
                    text = get_pdf_text(pdf_doc)
                    if text:
                        raw_text += text
                    else:
                        st.write(f"Failed to extract text from PDF: {pdf_doc.name}")

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.write(vectorstore)

                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
