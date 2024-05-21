import os
import logging
import streamlit as st
# import openai
import json
from dotenv import load_dotenv

from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient 

from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import Document
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.retrievers import AzureCognitiveSearchRetriever

from src.database_manager import DatabaseManager

load_dotenv()

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Disable logging for azure.core.pipeline.policies.http_logging_policy
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING)

# Azure Search configurations
AZURE_SEARCH_URL = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_URL")
AZURE_SEARCH_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME")
AZURE_SEARCH_KEY = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")

# Azure Storage configurations
STORAGE_CONTAINER_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
STORAGE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Form Recognizer configurations
FORM_RECOGNIZER_ENDPOINT = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY = os.getenv("FORM_RECOGNIZER_KEY")

# Azure OpenAI Chat Completion configurations
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Default items
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_CHAT_MODEL = "gpt-35-turbo"
DEFAULT_SEARCH_INDEX = "docindex"
DEFAULT_SEARCH_FILE_EXTENSION = ".pdf"
LOAD_VECTORS = True 

class EmbeddingPipeline:
    def __init__(self):
        load_dotenv()
        self.form_recognizer_client = self.get_form_recognizer_client()
        self.db_manager = DatabaseManager("db/metadata.db")
        self.embedder = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL,
                            openai_api_base=AZURE_OPENAI_ENDPOINT,
                            openai_api_key=AZURE_OPENAI_API_KEY ,
                            openai_api_version="2023-05-15",
                            openai_api_type="azure")
        # self.embedder = AzureOpenAIEmbeddings(
        #     azure_endpoint=AZURE_OPENAI_ENDPOINT ,
        #     azure_deployment=DEFAULT_EMBEDDING_MODEL,
        #     openai_api_version="2023-05-15"
        # )

    def get_form_recognizer_client(self):
        """
        Get an instance of the Form Recognizer Client.

        Returns:
            FormRecognizerClient: An instance of the FormRecognizerClient class.
        """
        credential = AzureKeyCredential(FORM_RECOGNIZER_KEY)
        return FormRecognizerClient(endpoint=FORM_RECOGNIZER_ENDPOINT, credential=credential)

    def form_recognizer_data_extract(self, blob_content):
        """
        Azure Form Recogniser extracts text from the PDF files loaded from the container. 
        NOTE: You can process other data files (.docx, .pptx etc) by manually converting them to .pdf as form recognizer doesnt support all the microsoft file types.

        Args:
            blob_content (bytes): The content of the blob to extract data from.

        Returns:
            tuple: A tuple containing:
                - list of dictionaries: Extracted table data.
                - list of dictionaries: Extracted line data.
                - str: Extracted text data.
        """
        table_data = []
        line_data = []
        text = ""

        try:
            form_recognizer_result = self.form_recognizer_client.begin_recognize_content(
                blob_content).result()

            for page in form_recognizer_result:
                for table in page.tables:
                    table_info = {"table_cells": []}
                    for cell in table.cells:
                        cell_info = {
                            "text": cell.text,
                            "bounding_box": cell.bounding_box,
                            "column_index": cell.column_index,
                            "row_index": cell.row_index
                        }
                        table_info["table_cells"].append(cell_info)
                    table_data.append(table_info)
                for line in page.lines:
                    text += " ".join([word.text for word in line.words]) + "\n"
                    line_info = {
                        "text": line.text,
                        "bounding_box": line.bounding_box
                    }
                    line_data.append(line_info)

            logger.info(
                "\t\tStep 3: Azure Form Recognizer - Extracted text from the file/s loaded from the container")

            return table_data, line_data, text

        except Exception as e:
            logger.warning(
                f"\t\tStep 3 (ERROR): Azure Form Recognizer - An error occurred while extracting form data: {e}")
            return [], [], []


    def get_text_chunks(self, text, blob_name):
        """
        Split a large text into smaller chunks for further processing.

        Args:
            text (str): The text to be split into chunks.

        Returns:
            list of Document: List of Document objects representing text chunks.
        """
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata = {"source":blob_name}) for chunk in chunks]
    
        logger.info("\t\tStep 4: Pre-Embedding - File is split into many smaller chunks")

        return docs


    def load_vectorstore(self, documents):
        """
        Azure OpenAI "text-embedding-ada-002" model prepare embeddings to the chunked files and upload vectors into Azure Cognitive Search Index.

        Args:
            documents (list of dict): List of documents to be added to Azure Cognitive Search.

        Returns:
            AzureSearch: An instance of AzureSearch containing the loaded vectors.
        """
        # try:
        vectorstore = AzureSearch(
            azure_search_endpoint=AZURE_SEARCH_URL,
            azure_search_key=AZURE_SEARCH_KEY,
            index_name=DEFAULT_SEARCH_INDEX,
            embedding_function=self.embedder.embed_query,
        )
        vectorstore.add_documents(documents=documents)
        logger.info(
            f"\t\tStep 5: Azure Cognitive Search - Embeddings are created and vectors are stored in Azure Search index: '{DEFAULT_SEARCH_INDEX}'")

        # except openai.error.APIError as api_error:
        #     logger.error(
        #         f"\t\tStep 5 (ERROR): Azure Cognitive Search - Error: {api_error}")


    def perform_embedding_pipeline(self):
        """
        Process documents in an Azure Storage container and perform an embedding pipeline on them.

        This function retrieves documents stored in an Azure Storage container specified by 'container_name'
        and processes each document. It checks if the document's content type matches a predefined extension 
        (e.g., '.pdf') and, if so, extracts data using Form Recognizer, processes the extracted data, 
        and loads it into a vector store.

        Parameters:
            storage_connection_string (str): The connection string for the Azure Storage account where the 
                container is located.
            container_name (str): The name of the Azure Storage container containing the documents to process.
        """
        logger.info( f"__NOTE__ Processing only {DEFAULT_SEARCH_FILE_EXTENSION} types")

        try:
            blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONTAINER_STRING)
            blob_container_client = blob_service_client.get_container_client(STORAGE_CONTAINER_NAME)
            exists, inserted = 0, 0
            for blob in blob_container_client.list_blobs():
                if blob.name.endswith(DEFAULT_SEARCH_FILE_EXTENSION) and not self.db_manager.record_exists_in_database(blob.name):
                    blob_client = blob_service_client.get_blob_client(container = STORAGE_CONTAINER_NAME, blob = blob.name)
                    blob_content = blob_client.download_blob().readall()  
                    logger.info( f"\tProcessing Document '{blob.name}' : ")
                    logger.info( f"\t\tStep 2: Azure Storage Container - Blob content fetched successfully")
                    
                    # only using 'raw_text' as of now
                    table_data, line_data, raw_text = self.form_recognizer_data_extract(blob_content)
                    documents = self.get_text_chunks(raw_text, blob.name)
                    self.load_vectorstore(documents) 
                    self.db_manager.insert_record_to_database(blob.name, 'Y', 'Y')
                    inserted +=1

                elif blob.name.endswith(DEFAULT_SEARCH_FILE_EXTENSION) and self.db_manager.record_exists_in_database(blob.name):
                    exists +=1

            logger.info(f"Embedding Summary : Processed {inserted} new file(s) out of {exists + inserted} total file(s) in the container. Stored vectors for {inserted} new file(s); as {exists} file(s) already had vectors.")
        except Exception as e:
            print(e)

class ChatPipeline:
    def __init__(self):
        load_dotenv()
        self.retriever = AzureCognitiveSearchRetriever(
                            content_key="content",
                            index_name=DEFAULT_SEARCH_INDEX,
                            service_name=AZURE_SEARCH_NAME,
                            api_key=AZURE_SEARCH_KEY,
                            top_k=1
                        )
        self.llm = AzureChatOpenAI(
                        deployment_name=DEFAULT_CHAT_MODEL,
                        openai_api_version="2023-05-15",
                        temperature=0,
                        openai_api_base=AZURE_OPENAI_ENDPOINT,
                        openai_api_key=AZURE_OPENAI_API_KEY 
                      )

    def get_query_answer(self, query, verbose=True):
        """
        Retrieve an answer from a question answering system that utilizes a pretrained language model (llm) 
        and a pre-defined retriever (Azure Cognitive Search) (retriever) to respond to questions following a predefined question answering chain (chain_type).

        Args:
            query (str): The query for which an answer is sought.
            retriever: The retriever used for retrieving relevant documents.
            verbose (bool, optional): Whether to log detailed information. Default is True.

        Returns:
            dict: A dictionary containing the query and the generated answer.
        """
        # Create a chain to answer questions
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type='stuff', 
            retriever=self.retriever, 
            return_source_documents=True
        )

        logger.info(f"Generating response ⏳")
        result = qa({"query": query})
        print(result)
        if verbose:
            logger.info(
                f"\n\nQ: {result['query']}\nA: {result['result']}\nSource/s: {set(json.loads(i.metadata['metadata']).get('source') for i in result['source_documents'])}")

        return result['result'], set(json.loads(i.metadata['metadata']).get('source') for i in result['source_documents'])

def initialize_azure_blob_client():
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONTAINER_STRING)
    return blob_service_client.get_container_client(STORAGE_CONTAINER_NAME)

def main():

    if LOAD_VECTORS:
        EmbeddingPipeline().perform_embedding_pipeline()
    else:
        logger.info(f"Retrieving the stored vectors from an Azure Search index: '{DEFAULT_SEARCH_INDEX}'")

    ### STREAMLIT UI

    st.set_page_config(page_title="my assistant")
    st.title("My Assistant")
    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])

    if uploaded_file is not None:
        container_client = initialize_azure_blob_client()
        
        # Upload the file to Azure Blob Storage
        blob_client = container_client.get_blob_client(uploaded_file.name)
        blob_client.upload_blob(uploaded_file)

        st.success(f"File uploaded successfully to Azure Blob Storage: {uploaded_file.name}")


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your query"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

        answer , source = ChatPipeline().get_query_answer(prompt)
        full_response += f"{answer} *(source: {', '.join(source)})*" if source and None not in source else f"{answer}"
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()