import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from transformers import BertModel, BertTokenizer
import altair as alt
import torch

OPENAI_API_KEY = "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"
#OPENAI_API_KEY = "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"
HUGGINGFACEHUB_API_TOKEN = "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" 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len   
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(preprocessed_text):
    #embeddings = OpenAIEmbeddings('bert-base-uncased')
    model_name = "bert-base-uncased"
    tokenizer= BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    tokens = tokenizer(preprocessed_text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state

    return embeddings


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY)
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":256},huggingfacehub_api_token="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")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
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
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text (example using whitespace tokenization)
    tokens = text.split()
    
    # Return the preprocessed tokens as a list or a formatted string
    return tokens


def main():
    load_dotenv()
    st.set_page_config(page_title="Trade Chat",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    styl = f"""
    <style>
        .stTextInput {{
        position: fixed;
        bottom: 3rem;
    }}
    </style>
    """

    st.markdown(styl, unsafe_allow_html=True)
    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Trade Chat :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                #text_chunks = get_text_chunks(raw_text)
                preprocessed_text = preprocess_text(raw_text)

                # create vector store
                embeddings = get_vectorstore(preprocessed_text)

                print(embeddings)
                # create conversation chain
                #st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()