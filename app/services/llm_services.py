# services/llm_services.py
import os
import json
from openai import AzureOpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.bedrock import Bedrock
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ['AWS_DEFAULT_REGION'] = 'ap-south-1'


class LLMService:
    def __init__(self, config_path='../../config.json'):
        """Initialize LLM service with configuration for both LLM and embedding models."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.llm = None
        self.initialize_llm()
        self.embed_model = AzureOpenAIEmbedding(
            model=self.config.get('embedding_model', 'text-embedding-ada-002'),
            deployment_name=self.config.get('embedding_deployment', 'text-embedding-ada-002'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION')
        )
        Settings.embed_model = self.embed_model

    def initialize_llm(self):
        """Choose LLM type and initialize based on configuration."""
        llm_type = self.config.get('llm_type', 'azure')
        if llm_type == 'azure':
            self.init_azure_openai()
        elif llm_type == 'bedrock':
            self.init_bedrock_llm()

    def init_azure_openai(self):
        """Initialize Azure OpenAI LLM."""
        self.llm = AzureOpenAI(
            engine=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            model=self.config.get('model', 'gpt-4o'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            temperature=self.config.get('temperature', 0.3),
            max_tokens=self.config.get('max_tokens', 1000),
            max_retries=self.config.get('max_retries', 3),
            system_prompt="Strictly avoid using sensitive, jailbreak, hate or offensive language."
        )
        Settings.llm = self.llm

    def init_bedrock_llm(self):
        """Initialize Bedrock LLM."""
        self.llm = Bedrock(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            aws_region_name='ap-south-1',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            temperature=self.config.get('temperature', 0.2),
        )
        Settings.llm = self.llm

    def get_llm(self):
        return self.llm

    def get_embed_model(self):
        return self.embed_model
