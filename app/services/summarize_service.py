import os
import getpass
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI  # Import the Google Generative AI class
from app.core.config import settings

class SummarizationService:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        self._set_environment_variables()
        self.client = self._initialize_client()

    def _set_environment_variables(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "True"
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass(settings.GEMINI_API_KEY)
        if "LANGCHAIN_SMITH_KEY" not in os.environ:
            os.environ["LANGCHAIN_SMITH_KEY"] = getpass.getpass(settings.LANGCHAIN_SMITH_KEY)

    def _initialize_client(self):
        return GoogleGenerativeAI(temperature=0, model="models/text-bison-001", api_key=settings.GEMINI_API_KEY)

    def _create_map_chain(self):
        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        return LLMChain(llm=self.client, prompt=map_prompt)

    def _create_reduce_chain(self):
        reduce_template = """The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        return LLMChain(llm=self.client, prompt=reduce_prompt)

    def _create_map_reduce_chain(self, map_chain, reduce_chain):
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )
        return MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

    def summarize(self, url: str) -> list:
        map_chain = self._create_map_chain()
        reduce_chain = self._create_reduce_chain()
        map_reduce_chain = self._create_map_reduce_chain(map_chain, reduce_chain)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=0
        )

        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            result = map_reduce_chain.invoke(split_docs)
            return result["output_text"]
        except Exception as e:
            if "400 The requested language is not supported by models/text-bison-001" in str(e):
                return "Error: The requested language is not supported by models/text-bison-001."
            else:
                raise e

summarization_service = SummarizationService()
