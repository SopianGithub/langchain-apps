import os
import getpass
import groq
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

class GROQService:
    def __init__(self):
        self._set_groq_api_key()
        self.client = groq.Client(api_key=settings.GROQ_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        self.pcindex = self._initialize_pinecone()

    def _set_groq_api_key(self):
        if "GROQ_API_KEY" not in os.environ:
            os.environ["GROQ_API_KEY"] = getpass.getpass(settings.GROQ_API_KEY)

    def _initialize_pinecone(self):
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        return pc.Index(settings.PINECONE_API_INDEX)

    def process_data(self, data: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": data}],
                model="gemma-7b-it"
            )
            return chat_completion.choices[0].message.content
        except groq.GROQError as e:
            raise Exception(f"Error processing data: {e}")

    def search_csv(self, query: str) -> list:
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = self._adjust_embedding_dimension(query_embedding)
        response = self.pcindex.query(query_embedding, top_k=5, include_metadata=True)
        return [match['metadata']['text'] for match in response['matches']]

    def _adjust_embedding_dimension(self, embedding: list) -> list:
        index_dimension = 1536  # The dimension of the Pinecone index
        if len(embedding) != index_dimension:
            if len(embedding) < index_dimension:
                embedding += [0] * (index_dimension - len(embedding))
            else:
                embedding = embedding[:index_dimension]
        return embedding

    def chat_with_grok(self, query: str) -> str:
        context_texts = self.search_csv(query)
        combined_query = self._create_combined_query(context_texts, query)
        return self._get_chat_response(combined_query)

    def _create_combined_query(self, context_texts: list, query: str) -> str:
        context = " ".join(context_texts)
        return f"Context: {context}\nUser: {query}"

    def _get_chat_response(self, combined_query: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": combined_query}],
                model="gemma-7b-it"
            )
            return chat_completion.choices[0].message.content
        except groq.GROQError as e:
            raise Exception(f"Error getting chat response: {e}")

    def summarize_text(self, text: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": text}],
                model="gemma-7b-it"
            )
            return chat_completion.choices[0].message.content
        except groq.GROQError as e:
            raise Exception(f"Error summarizing text: {e}")

groq_service = GROQService()
