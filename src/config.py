import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    MODEL_NAME: str
    EMBEDDINGS_MODEL_NAME: str
    API_CLIENT_TOKEN: str
    QDRANT_ADDRESS: str
    QDRANT_PORT: str
    QDRANT_COLLECTION: str
    RAG_SERVICE_ADDRESS: str
    RAG_SERVICE_PORT: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"),
        extra="ignore",
    )

    def get_model_name(self):
        return self.MODEL_NAME

    def get_embeddings_model_name(self):
        return self.EMBEDDINGS_MODEL_NAME
    
    def get_model_key(self):
        return self.API_CLIENT_TOKEN

    def get_qdrant_url(self):
        return f"http://{self.QDRANT_ADDRESS}:{self.QDRANT_PORT}"

    def get_qdrant_collection(self):
        return self.QDRANT_COLLECTION
    
    def get_rag_service_url(self):
        return f"http://{self.RAG_SERVICE_ADDRESS}:{self.RAG_SERVICE_PORT}/"


settings = ModelSettings()
