import streamlit as st
import yaml
import requests
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

def yaml_to_chunks(yaml_data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    yaml_string = yaml.dump(yaml_data)  # Convert YAML to string for chunking
    chunks = text_splitter.split_text(yaml_string)  # Split the YAML string into chunks
    return chunks

# Function to generate cURL command from API specs
def generate_curl(api_spec):
    method = api_spec.get('method', 'GET')
    url = api_spec.get('url', '')
    headers = api_spec.get('headers', {})
    data = api_spec.get('data', {})

    # Construct cURL command
    curl_cmd = f"curl -X {method.upper()} '{url}'"

    # Add headers
    for header, value in headers.items():
        curl_cmd += f" -H '{header}: {value}'"

    # Add data
    if data:
        curl_cmd += f" -d '{data}'"

    return curl_cmd



OPENAI_API_KEY = "dummy"
st.header("My first chatbot")
with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a yml file and start asking questions", type="yml")

#Extract the text
if file is not None:
    # Parse the uploaded YAML file
    api_specs = yaml.safe_load(file)

    #st.write("Parsed API Specs:")
    #st.write(api_specs)

    # Extract API specs and generate cURL commands
    # if isinstance(api_specs, list):  # If the YAML file has a list of API specs
    #     for api_spec in api_specs:
    #         curl_command = generate_curl(api_spec)
    #         #st.code(curl_command)  # Display the cURL command
    # else:  # If the YAML file contains a single API spec
    #     curl_command = generate_curl(api_specs)
        #st.code(curl_command)



    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # creating vector store - FAISS
    vector_store = FAISS.from_texts(yaml_to_chunks(api_specs), embeddings)
    # get user question
    user_question = st.text_input("Type your question here")
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define the llm
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)



