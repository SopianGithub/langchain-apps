import os
import getpass
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class NERExtractionService:
    def __init__(self, model="models/text-bison-001"):
        self._set_google_api_key()
        self.client = self._initialize_google_client(model)
        self.prompt_template = self._create_prompt_template()
        self.ner_chain = self._load_ner_chain()

    def _set_google_api_key(self):
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

    def _initialize_google_client(self, model):
        return GoogleGenerativeAI(api_key=os.environ["GOOGLE_API_KEY"], model=model)

    def _create_prompt_template(self):
        return PromptTemplate(
            input_variables=["text"],
            template="Extract named entities from the following text:\n\n{text}\n\nEntities:"
        )

    def _load_ner_chain(self):
        return LLMChain(llm=self.client, prompt=self.prompt_template)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text

    def extract_entities(self, text: str) -> str:
        try:
            result = self.ner_chain.run({"text": text})
            return result
        except Exception as e:
            if "400 The requested language is not supported by models/text-bison-001" in str(e):
                return "Error: The requested language is not supported by models/text-bison-001."
            else:
                raise e
            
ner_services = NERExtractionService()