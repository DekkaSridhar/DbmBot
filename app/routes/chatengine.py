import os
import warnings
import asyncio
import logging
from services.llm_services import LLMService
from services.docblob import DocumentEnhancer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.memory import SimpleComposableMemory, ChatMemoryBuffer
from dotenv import load_dotenv
from docling.datamodel.pipeline_options import AcceleratorDevice
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
import nest_asyncio

# Apply nest_asyncio for Jupyter and other environments
nest_asyncio.apply()

# Suppress cryptography deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tools:
    def __init__(self):
        logger.debug("Initializing Tools class...")
        # Load environment variables
        load_dotenv(override=True)
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

        # Initialize the LLM and embedding model
        self.llm_initializer = LLMService('config.json')
        self.llm = self.llm_initializer.get_llm()
        self.embed_model = self.llm_initializer.get_embed_model()

        # Postprocessor for metadata
        self.postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

        # Persistent directory for index storage
        self.persist_dir = "dbmindexstore"
        # Initialize index
        self.index = asyncio.run(self._initialize_index())

    async def _initialize_index(self):
        try:
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                logger.info("Loading existing index...")
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                return load_index_from_storage(storage_context)
            else:
                logger.info("Creating a new index from documents...")
                # local_directory = "docs"  # Replace with your directory containing documents
                enhancer = DocumentEnhancer(self.connection_string, self.container_name)
                enhanced_documents = enhancer.load_and_enhance_documents()

                # Load and process documents
                documents = [Document(text=doc.text, metadata=doc.metadata) for doc in enhanced_documents]
                logger.debug("Documents loaded and enhanced.")

                node_parser = MarkdownNodeParser()
                self.nodes = node_parser.get_nodes_from_documents(documents)
                logger.debug("Nodes parsed from documents.")

                extractors = [
                    SummaryExtractor(summaries=["prev", "self", "next"], llm=self.llm, num_workers=4),
                    QuestionsAnsweredExtractor(questions=3, llm=self.llm, metadata_mode=MetadataMode.EMBED, num_workers=4),
                ]

                # Parallelize pipeline processing
                pipeline = IngestionPipeline(transformations=[node_parser, *extractors])
                nodes1 = await pipeline.arun(nodes=self.nodes, in_place=False, show_progress=True)

                # Create and save the index
                index = VectorStoreIndex(self.nodes + nodes1, use_async=True)
                logger.info("Index created and nodes ingested.")

                index.storage_context.persist(persist_dir=self.persist_dir)
                logger.debug("Index persisted to storage.")
                return index
        except Exception as e:
            logger.error(f"Error initializing index: {e}", exc_info=True)

    def query_document_tool(self, query: str):
        try:
            logger.debug(f"Received query: {query}")
            engine = self.index.as_chat_engine(
                similarity_top_k=4,
                query_transform=HyDEQueryTransform(llm=self.llm, include_original=True),
                node_postprocessors=[self.postproc],
                memory=SimpleComposableMemory.from_defaults(
                    primary_memory=ChatMemoryBuffer.from_defaults(
                        chat_history=[], token_limit=3000, chat_store=SimpleChatStore(), chat_store_key="user1"
                    )
                ),
                chat_mode="context",
                system_prompt="""
                You are a Document Analysis Assistant specializing in extracting and explaining information strictly based on the provided document. Your primary role is to retrieve, analyze, and explain relevant sections and clauses while explicitly referencing section and rule numbers in your responses.
                Understanding the Query:
                -Carefully analyze the user's query to ensure clarity and specificity.
                -If the query is ambiguous or incomplete, reframe or optimize it for precision.
                -Always align the query with the document's provided context.

                Retrieval and Analysis:
                -Search only within the provided document and its associated context.
                -Every answer must explicitly reference the section number and rule number of the relevant text.
                -Retrieve sections or clauses verbatim from the document that directly or indirectly address the query.
                -Cross-reference related sections for completeness when needed.

                Output Requirements:
                -Answers: Give Descriptive answers whenever required. Answers must contain section number and rule number references.Ask if user want more informtion whenever necessary.
                -Professional Tone: Adopt a tone similar to that of a "Secretary of Budget", ensuring formality and directness. Avoid excessive use of “document” references.
                -Circulars and Lists: If the query pertains to a circular or list, provide only the location or name of the requested items. Avoid unnecessary elaboration.
                -Charts and Graphs: For queries about trends, comparisons, or quantitative data, include charts or visual representations to clarify the analysis.
                -Icons and Emojis: Where applicable, include icons or emojis to highlight important points and enhance readability.
                -Step-by-Step Guidance: Provide step-by-step guidance for complex queries, ensuring clarity and ease of understanding.

                """
            )

            response = engine.chat(query)
            logger.info("Query processed successfully.")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)

def chatbot_loop():
    try:
        tools = Tools()
        print("Welcome to the Procurement Chatbot!")
        print("Ask me questions about the 2016 Revised IRR of Republic Act No. 9184.")
        print("Type 'exit' to end the conversation.\n")

        while True:
            query = input("Your query: ")
            if query.lower() == "exit":
                print("Thank you for using the chatbot. Goodbye!")
                break

            response = tools.query_document_tool(query)
            print("\nChatbot Response:")
            print(response)
            print("\n")
    except Exception as e:
        logger.critical(f"Critical error in chatbot loop: {e}", exc_info=True)

# Run the chatbot loop
# asyncio.run(chatbot_loop())
