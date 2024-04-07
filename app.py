from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit_scrollable_textbox as stx
from langchain.vectorstores.chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader


@st.cache_resource
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_resource
def vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=text_chunks , embedding= embeddings , persist_directory="/Users/sid/Documents/Computer Vision/MedGPT/data")
    return vectorstore

@st.cache_resource
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

@st.cache_resource
def load_model():
    return HuggingFaceEndpoint(
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature = 0.1,
        top_p = 0.9,
        return_full_text = False,
        model_kwargs = {
            "max_length" : 1024,
        } 
    )

def generate_response(ensemble_retriever , query):
    filtered_docs = ensemble_retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in filtered_docs])
    hub_llm = load_model()
    template = """
    User: You are an AI Assistant that follows instructions extremely well.
    Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in context
    Keep in mind, you will lose the job, if you answer out of context questions
    context: {context}
    query: {question}
    Remember only return AI answer
    Assistant:
    """

    updated_template = template.format(context = context , question = query) 
    prompt = ChatPromptTemplate.from_template(updated_template)
    hub_chain = LLMChain(prompt = prompt , llm = hub_llm)
    response = hub_chain.invoke({"context" : context , "query" : query})
    return response


def main():

    st.set_page_config(page_title="Chat with PDFs" , page_icon=":boooks")

    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) LLM model''')
        add_vertical_space(5)
        st.write('Made with ❤️ by [Sidharth](https://sidharthsingh.site)')

    st.header("Chat with PDF :books:")

    pdf_docs = st.file_uploader("Upload your PDF 😎" , type='pdf' , accept_multiple_files=True)

    if pdf_docs is not None and len(pdf_docs) > 0:
        load_dotenv()

        raw_text = get_pdf_text(pdf_docs)

        text_chunks = get_text_chunks(raw_text)

        vector_db = vector_store(text_chunks=text_chunks)

        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 4})

        keyword_retriever = BM25Retriever.from_texts(text_chunks)
        keyword_retriever.k = 2

        ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever , keyword_retriever] , weights=[0.8 , 0.2])
        
        query = st.text_input("What do you wanna know about your PDF 😊")
        if query:
            response = generate_response(ensemble_retriever=ensemble_retriever , query=query)
            formatted_response = response['text'].split('\n')
            stx.scrollableTextbox(text='\n'.join(formatted_response) , height= 400)
            st.subheader("Here is the context")
            stx.scrollableTextbox(text='\n'.join(response['context'].split('\n')) , height= 400)
        


if __name__ == '__main__':
    main()