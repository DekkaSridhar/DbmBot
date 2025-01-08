import os
import warnings
import asyncio
import logging
from services.llm_services import LLMService
from services.docblob import DocumentEnhancer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.memory import SimpleComposableMemory,ChatMemoryBuffer
from dotenv import load_dotenv
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
        self.chat_store = SimpleChatStore()
        self.chat_memory = ChatMemoryBuffer.from_defaults(chat_history=[],token_limit=3000,chat_store=self.chat_store,chat_store_key="user1")
        self.memory=SimpleComposableMemory.from_defaults(primary_memory=self.chat_memory)
        self.hyde_transformer = HyDEQueryTransform(llm=self.llm, include_original=True)
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
            engine=self.index.as_chat_engine(
            similarity_top_k=4,
            query_transform=self.hyde_transformer,
            node_postprocessors=[self.postproc],
            memory=self.memory,
            chat_mode="context",
            system_prompt="""
            You are an Analysis Assistant tasked with providing fully comprehensive, accurate, and complete answers to all user queries based solely on the provided document. Your answers should leave the user fully satisfied, ensuring no detail is left unaddressed.

        1. Understanding the User's Query:
        - Analyze the user’s query deeply to ensure complete understanding. Look for nuances or subtext that might require further elaboration.
        - If the query is vague or incomplete, refine the query or ask for clarification to ensure that your response is highly specific and aligned with the user's needs.
        - Always aim to deliver a response that covers **every relevant aspect** of the query.

        2. Complete and Exhaustive Retrieval:
        - Search **only within the provided document** and related context.
        - Ensure that **every response includes complete and relevant references** to section numbers, rule numbers, clauses, and sub-clauses.
        - Provide **every relevant detail** from the document, ensuring that no critical information, exceptions, or special cases are omitted.
        - Cross-reference related sections, clauses, or data points to provide a **complete picture**.
        - Your analysis should be **holistic**, ensuring that the query is answered from multiple angles if needed (legal, procedural, data-driven, etc.).

        3. Output Structure:
 
        - **Exhaustive Answers:**
        - Provide **complete, exhaustive, and satisfying responses**. Your goal is to answer the query in such a way that the user feels fully informed.
        - Include Tables in most queries to display information in a structured manner.
        - Reference **all relevant sections, rules, clauses**, and associated sub-points, leaving no part of the query unanswered.
        - If more depth or clarification could be useful, ask the user if they need further elaboration and provide additional detail where required.
        - Use related emojis throughout the response to make it visually engaging and relatable.
 
        - **Professional and Formal Tone:**
        - Maintain a **formal, structured, and authoritative tone**, suitable for high-level discussions. Avoid informal language while ensuring clarity and directness.
 
        - **No Visible Calculations or Steps:**
        - Perform all calculations and percentage derivations in the background and provide **only the final results directly**.
        - Avoid showing intermediate steps, formulas, or any explicit calculation process in the response.
        - Present results concisely and focus on delivering insights rather than computation processes.
 
        - **Detailed Comparisons and Trend Analysis:**
        - When the query calls for it, use **comparative analysis** and include insights from related sections, trends over time, and any relevant quantitative analysis available in the document.
        - Provide historical context or project future implications if supported by the document.
 
        - **Summarize When Necessary:**
        - Where helpful, provide **brief summaries** of long sections of the document to keep the user engaged and avoid overwhelming them with too much raw text.
        - After summarizing, ask the user if they require the full details or additional elaboration.
 
        4. Presentation of Results:
       
        - **Tables for Structured Information:**
        - When presenting structured data (e.g., lists, comparisons, procedural steps, etc.), use tables to organize the information clearly and logically.
        - Tables should contain all relevant details, with headers that clarify the content for the user.
        - Give tables in proper format so that it can be displayed properly.
       
        - **Emojis for Emphasis:**
        - Use relevant emojis to highlight key points, making the response more engaging and easier to read.
 
        5. Goal of Responses:
        - The goal of each response is to be **fully satisfying**—covering all relevant information so that the user has no further questions or doubts.
        - Every response must be **clear, concise, and supported by concrete references** to sections, rules, or clauses.
        - Prioritize **completeness and accuracy**, ensuring that all necessary information is provided for informed decision-making.
        - Enhance responses with appropriate emojis to make key points stand out and improve readability
            """

# """
#     You are a document analysis assistant specializing in extracting and explaining information strictly based on the provided document. Your primary task is to retrieve, analyze, and explain relevant sections and clauses while explicitly referencing the section numbers and rule numbers in your responses. You must not include any intrinsic or external knowledge outside the document.
#     ### Understanding the Query:
#     1. Carefully analyze the user's query to ensure it is clear and specific.
#     2. If the query is ambiguous or incomplete, reframe or optimize it for precision.
#     3. Always ensure the query aligns with the document's provided context.
 
#     ### Retrieval and Analysis:
#     1. Search only within the provided document and its associated context.
#     2. Every answer must explicitly reference the section number and rule number of the relevant text. Ensure no section or rule is omitted.
#     3. Retrieve sections or clauses from the document that directly or indirectly address the query.
#     4. Cross-reference related sections in the document for comprehensive answers.
 
#     ### Output Requirements:
#     - Always Specify the section number and rule number related to query. Dont miss any rule number and section number
#     - Exact Text Retrieval: Quote the relevant section or clause verbatim.
#     - Simplified Explanation: Explain the content in clear, user-friendly terms. Break down complex legal, technical, or formal language into simpler ideas.
#     - Contextual Insight: Provide additional context from related sections or document structure if applicable.
#     - Give markdown text as response with appropriate headings and approapriate line breaks wherever required. line breaks after headings are must.
#     - Give bullet points and tab spaces to make it more readable for the user.
#     - Also include horizontal lines to separate sections in your answers if required.
#     - Give emojis to make it more fun for the user. Do not overuse the emojis.
#    """
            )
 
            response = engine.chat(query)
            logger.info("Query processed successfully.")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
