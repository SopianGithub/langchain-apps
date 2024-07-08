import os
import getpass
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from app.core.config import settings
import pandas as pd

class GenAIService:
    def __init__(self):
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass(settings.GEMINI_API_KEY)
        
        self.chat_client = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.pcindex = pc.Index(settings.PINECONE_API_INDEX)

    def load_csv(self, csv_data: pd.DataFrame):
        if self.check_csv_upsert(csv_data) is False:
            texts = [" ".join(map(str, row.values)) for index, row in csv_data.iterrows()]
            index_dimension = 1536  # The dimension of the Pinecone index
            for i, text in enumerate(texts):
                embedding = self.embeddings.embed_query(text)
                if len(embedding) != index_dimension:
                    # Adjust the embedding dimension to match the index dimension
                    if len(embedding) < index_dimension:
                        # Pad the embedding with zeros if it's smaller
                        embedding = embedding + [0] * (index_dimension - len(embedding))
                    else:
                        # Truncate the embedding if it's larger
                        embedding = embedding[:index_dimension]
                self.pcindex.upsert([(str(i), embedding, {"text": text})])
            return texts
        else:
            return 'Has Upsert'
    
    def check_csv_upsert(self, csv_data: pd.DataFrame) -> bool:
        texts = [" ".join(map(str, row.values)) for index, row in csv_data.iterrows()]
        for i, text in enumerate(texts):
            response = self.pcindex.fetch(ids=[str(i)])
            if not response or str(i) not in response['vectors']:
                return False
        return True
    
    def search_csv(self, query: str):
        query_embedding = self.embeddings.embed_query(query)
        index_dimension = 1536  # The dimension of the Pinecone index
        if len(query_embedding) != index_dimension:
            # Adjust the embedding dimension to match the index dimension
            if len(query_embedding) < index_dimension:
                # Pad the embedding with zeros if it's smaller
                query_embedding = query_embedding + [0] * (index_dimension - len(query_embedding))
            else:
                # Truncate the embedding if it's larger
                query_embedding = query_embedding[:index_dimension]

        response = self.pcindex.query(query_embedding, top_k=5, include_metadata=True)
        return [match['metadata']['text'] for match in response['matches']]

    def chat_with_genai(self, query: str):
        context_texts = self.search_csv(query)
        context = " ".join(context_texts)
        combined_query = f"Context: {context}\nUser: {query}"
        human_message = HumanMessage(content=combined_query)
        # Fix the invalid input type error by converting HumanMessage to a list of BaseMessages
        response = self.chat_client.invoke([human_message])
        return response

    # def _cosine_similarity(self, vec1, vec2):
    #     dot_product = sum(p*q for p, q in zip(vec1, vec2))
    #     magnitude = (sum([val**2 for val in vec1]) * sum([val**2 for val in vec2])) ** 0.5
    #     if not magnitude:
    #         return 0
    #     return dot_product / magnitude
    
genai_service = GenAIService()