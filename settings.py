from pydantic_settings import BaseSettings
from pydantic import SecretStr
from pydantic.fields import Field

class OpenSearchSettings(BaseSettings):
    host: str = Field(validation_alias="OPENSEARCH_HOST")
    port: int = Field(validation_alias="OPENSEARCH_PORT")
    user: str = Field(validation_alias="OPENSEARCH_USER")
    password: SecretStr = Field(validation_alias="OPENSEARCH_PASSWORD")
    use_ssl: bool = Field(default=False, validation_alias="OPENSEARCH_USE_SSL")
    verify_certs: bool = Field(default=False, validation_alias="OPENSEARCH_VERIFY_CERTS")
    ssl_assert_hostname: bool = Field(default=False, validation_alias="OPENSEARCH_SSL_ASSERT_HOSTNAME")
    ssl_show_warn: bool = Field(default=False, validation_alias="OPENSEARCH_SSL_SHOW_WARN")
    index_name: str = Field(default="langchain_embeddings", validation_alias="OPENSEARCH_INDEX_NAME")
    embedding_model_name: str = Field(default="sentence-transformers/all-mpnet-base-v2", validation_alias="EMBEDDING_MODEL_NAME")
    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }
