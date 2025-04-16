"""
RAG Product Replacement Workflow

This script demonstrates a complete workflow for handling product replacements
in a RAG (Retrieval Augmented Generation) system. It shows how to:

1. Identify and extract content about a retired product
2. Update the vector database to reflect product retirement
3. Add new content for the replacement product
4. Implement retrieval logic that handles queries about both products
5. Support multiple LLM providers (OpenAI and local Ollama models)

Requirements:
- langchain-core
- langchain-community
- langchain-text-splitters
- langchain-chroma>=0.2.3
- chromadb (or other vector store)
- requests (for Ollama API)

One of the following is required for LLM functionality:
- langchain-openai and valid OPENAI_API_KEY (for OpenAI LLM and embeddings)
- Ollama running locally on port 11434 (for local LLM inference)

Usage:
python rag-update-demo.py [options]

Options:
  --vector-db PATH, -v PATH   Specify custom vector database path (default: ./demo_vector_db)
  --clean, -c                 Clean existing vector database before running
  --debug, -d                 Enable additional debug output
  
Note: If you encounter dimensionality errors with an existing vector database,
use the --clean flag to start with a fresh database.
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Vector database and embedding dependencies
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

# LLM components for RAG
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import BaseMessage, AIMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMConfig:
    """Configuration for LLM providers"""
    
    # Ollama configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2"  # Default model
    
    # OpenAI configuration
    OPENAI_MODEL = "gpt-3.5-turbo"

class OllamaChatLLM(BaseChatModel):
    """Ollama local LLM provider that implements the LangChain interface"""
    
    def __init__(self, model: str = LLMConfig.OLLAMA_MODEL, base_url: str = LLMConfig.OLLAMA_BASE_URL):
        """Initialize the Ollama chat model."""
        super().__init__()
        self.name = "Ollama"
        self.model = model
        self.base_url = base_url
        self._validate()
        
    def _validate(self):
        """Validate Ollama server connection"""
        try:
            # Check if Ollama server is running and model is available
            response = requests.get(f"{self.base_url}/api/tags")
            
            if response.status_code != 200:
                logger.warning(f"Ollama server returned status code {response.status_code}")
                return False
            
            # Check if our model is available
            available_models = response.json().get("models", [])
            model_names = [model.get("name") for model in available_models]
            
            if not model_names:
                logger.warning("No models found on Ollama server")
                return False
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available models: {', '.join(model_names)}")
                logger.info(f"Using {model_names[0]} instead.")
                self.model = model_names[0]
            
            logger.info(f"Ollama server validated successfully with model: {self.model}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error connecting to Ollama server: {str(e)}")
            logger.info("Make sure Ollama is running locally on port 11434")
            return False
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response using Ollama API"""
        try:
            # Extract the system message if present
            system_message = ""
            for message in messages:
                if message.type == "system":
                    system_message = message.content
                    break
            
            # Get the user's message (last one if multiple)
            last_message = messages[-1].content
                        
            payload = {
                "model": self.model,
                "prompt": last_message,
                "system": system_message or "You are a helpful assistant.",
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Ollama server returned status code {response.status_code}")
                
            result = response.json()
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=result.get("response", "").strip()))]
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            # Return a minimally valid result with an error message
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(
                    content=f"Error generating response from Ollama: {str(e)}"
                ))]
            )
    
    @property
    def _llm_type(self):
        """Return the LLM type identifier"""
        return "ollama-chat-llm"

def create_llm() -> BaseChatModel:
    """
    Create and return an LLM based on available providers.
    Tries OpenAI first, then Ollama. Raises an exception if neither is available.
    
    Returns:
        BaseChatModel: A LangChain compatible chat model
        
    Raises:
        Exception: If neither OpenAI nor Ollama is available
    """
    # Try to use OpenAI's ChatGPT if available
    try:
        if "OPENAI_API_KEY" in os.environ:
            try:
                # Initialize the LLM
                llm = ChatOpenAI(model=LLMConfig.OPENAI_MODEL, temperature=0)
                
                # Test with a simple query to verify API key validity and quota
                test_response = llm.invoke("This is a test message to verify API key. Reply with 'valid'.")
                if "valid" in test_response.content.lower():
                    logger.info(f"Using OpenAI {LLMConfig.OPENAI_MODEL} for text generation")
                    return llm
                else:
                    logger.warning("OpenAI API test response did not contain expected output")
                    raise ValueError("OpenAI API returned unexpected response")
                    
            except Exception as openai_err:
                if "429" in str(openai_err) or "quota" in str(openai_err).lower() or "insufficient_quota" in str(openai_err):
                    logger.error("OpenAI API quota exceeded or rate limit hit. Please check your billing details.")
                    raise ValueError(f"OpenAI API quota exceeded. Check your billing details at https://platform.openai.com/account/billing. Error: {str(openai_err)}")
                else:
                    logger.warning(f"OpenAI initialization error: {str(openai_err)}")
                    raise ValueError(f"OpenAI API error: {str(openai_err)}")
        else:
            logger.info("No OpenAI API key found in environment variables")
            raise ValueError("No OpenAI API key found in environment variables")
    except Exception as e:
        logger.warning(f"Could not initialize OpenAI: {str(e)}")
    
    # Try to use Ollama if available
    try:
        ollama_llm = OllamaChatLLM()
        # Check if Ollama is working
        if ollama_llm._validate():
            logger.info(f"Using Ollama with model {ollama_llm.model} for text generation")
            return ollama_llm
        else:
            logger.warning("Ollama validation failed")
            raise ValueError("Ollama validation failed")
    except Exception as e:
        logger.warning(f"Could not initialize Ollama: {str(e)}")
    
    # If we get here, neither OpenAI nor Ollama is available
    error_message = """
No LLM provider available. Please configure one of the following:
1. Set OPENAI_API_KEY environment variable with a valid API key and ensure you have quota available
2. Start a local Ollama server (http://localhost:11434) with at least one model installed
"""
    logger.error(error_message)
    raise Exception(error_message)

class ProductKnowledgeManager:
    """
    Manages product knowledge in a RAG system, handling the lifecycle of product information
    including retirement and replacement.
    """
    
    def __init__(
        self, 
        vector_db_path: str = "./chroma_db",
        embedding_model: Optional[Any] = None,
        text_splitter: Optional[Any] = None
    ):
        """
        Initialize the product knowledge manager.
        
        Args:
            vector_db_path: Directory path where the vector database is stored
            embedding_model: The embedding model to use (defaults to a simple mock embedding model)
            text_splitter: The text splitter to use for chunking documents
        """
        self.vector_db_path = vector_db_path
        
        # Initialize embedding model - use a simple mock if none provided and OpenAI not configured
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            # Try to use OpenAI embeddings if API key is available
            try:
                import os
                if "OPENAI_API_KEY" in os.environ:
                    from langchain_openai import OpenAIEmbeddings
                    try:
                        # Test the embeddings with a simple query
                        test_embeddings = OpenAIEmbeddings()
                        test_result = test_embeddings.embed_query("Test query for API validation")
                        if test_result and len(test_result) > 0:
                            self.embedding_model = test_embeddings
                            logger.info("Using OpenAI embeddings - API key validated successfully")
                        else:
                            raise ValueError("OpenAI embeddings returned empty result")
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e).lower() or "insufficient_quota" in str(e):
                            logger.error("OpenAI API quota exceeded or rate limit hit. Please check your billing details.")
                            raise ValueError("OpenAI API quota exceeded. Check your billing details at https://platform.openai.com/account/billing")
                        else:
                            logger.warning(f"OpenAI embeddings error: {str(e)}")
                            raise ValueError(f"OpenAI API error: {str(e)}")
                else:
                    raise ValueError("No OpenAI API key found")
            except (ImportError, ValueError) as e:
                # Fall back to a simple mock embedding model for demonstration
                from langchain_core.embeddings import Embeddings
                logger.info(f"Using mock embeddings: {str(e)}")
                
                class MockEmbeddings(Embeddings):
                    """Simple mock embedding model for demonstration purposes."""
                    def embed_documents(self, texts):
                        # Return random vectors of length 384 for each text
                        import numpy as np
                        return [np.random.rand(384).tolist() for _ in texts]
                    
                    def embed_query(self, text):
                        # Return a random vector of length 384
                        import numpy as np
                        return np.random.rand(384).tolist()
                
                self.embedding_model = MockEmbeddings()
        
        # Initialize text splitter with default parameters
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize or load the vector database
        self._initialize_vector_db()
        
        # Product relationship tracking
        self.product_relationships_file = os.path.join(vector_db_path, "product_relationships.json")
        self.product_relationships = self._load_product_relationships()
    
    def _check_and_repair_vector_db(self) -> bool:
        """
        Check if the vector database has dimension mismatch or is corrupted,
        and try to fix it by removing the database files.
        
        Returns:
            bool: True if repair was needed and performed, False otherwise
        """
        # Check if vector database exists
        if not os.path.exists(self.vector_db_path):
            return False
            
        try:
            # Try to load Chroma to see if it works
            test_db = Chroma(
                persist_directory=self.vector_db_path,
                collection_name="product_knowledge",
                embedding_function=self.embedding_model
            )
            
            # Try a basic operation to validate the database
            try:
                test_db._collection.count()
                # If we get here, the database is working fine
                return False
            except Exception as e:
                error_str = str(e).lower()
                if "dimension" in error_str or "does not match" in error_str:
                    logger.warning(f"Detected dimension mismatch in vector database: {str(e)}")
                    logger.warning(f"Removing vector database at {self.vector_db_path}")
                    
                    # Remove the entire directory
                    import shutil
                    shutil.rmtree(self.vector_db_path, ignore_errors=True)
                    
                    # Recreate an empty directory
                    os.makedirs(self.vector_db_path, exist_ok=True)
                    return True
                    
        except Exception as e:
            logger.warning(f"Vector database check failed: {str(e)}")
            # Only repair if it seems to be a database issue
            error_str = str(e).lower()
            if any(term in error_str for term in ["dimension", "corrupt", "invalid", "type", "schema"]):
                logger.warning(f"Removing potentially corrupted vector database at {self.vector_db_path}")
                
                # Remove the entire directory
                import shutil
                shutil.rmtree(self.vector_db_path, ignore_errors=True)
                
                # Recreate an empty directory
                os.makedirs(self.vector_db_path, exist_ok=True)
                return True
        
        return False
    
    def _initialize_vector_db(self) -> None:
        """Initialize or load the vector database from disk."""
        logger.info(f"Initializing vector database at {self.vector_db_path}")
        
        # Create the directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Check for corrupted database and repair if needed
        repaired = self._check_and_repair_vector_db()
        if repaired:
            logger.info("Vector database was repaired - starting with fresh database")
        
        try:
            # Try with collection_name parameter (newer versions might require it)
            try:
                # Initialize Chroma with persistence and collection name
                self.vectorstore = Chroma(
                    persist_directory=self.vector_db_path,
                    collection_name="product_knowledge",
                    embedding_function=self.embedding_model
                )
            except Exception as inner_e:
                logger.warning(f"Could not initialize with collection_name, trying without: {str(inner_e)}")
                # Try without collection_name as fallback
                self.vectorstore = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embedding_model
                )
            
            # Get collection count if possible
            collection_count = 0
            try:
                if hasattr(self.vectorstore, "_collection") and hasattr(self.vectorstore._collection, "count"):
                    collection_count = self.vectorstore._collection.count()
                elif hasattr(self.vectorstore, "get") and callable(getattr(self.vectorstore, "get")):
                    # Newer versions might use get() to retrieve all documents
                    results = self.vectorstore.get()
                    if results and isinstance(results, dict) and "ids" in results:
                        collection_count = len(results["ids"])
                logger.info(f"Vector database initialized with approximately {collection_count} documents")
            except Exception as count_e:
                logger.warning(f"Could not get collection count: {str(count_e)}")
                logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Chroma: {str(e)}")
            raise Exception(f"Could not initialize vector database: {str(e)}")
    
    def _load_product_relationships(self) -> Dict:
        """Load product relationships from disk or initialize if not present."""
        if os.path.exists(self.product_relationships_file):
            with open(self.product_relationships_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize with empty relationship structure
            relationships = {
                "products": {},
                "replacements": {},
                "history": []
            }
            self._save_product_relationships(relationships)
            return relationships
    
    def _save_product_relationships(self, relationships: Dict = None) -> None:
        """Save product relationships to disk."""
        with open(self.product_relationships_file, 'w') as f:
            json.dump(relationships or self.product_relationships, f, indent=2)
        logger.info(f"Product relationships saved to {self.product_relationships_file}")
    
    def identify_product_documents(self, product_id: str) -> List[Tuple[str, Dict]]:
        """
        Identify all documents in the vector store related to a specific product.
        
        Args:
            product_id: The ID of the product to search for
            
        Returns:
            List of tuples containing document IDs and their metadata
        """
        try:
            # Try newer API approach first
            if hasattr(self.vectorstore, "get") and callable(getattr(self.vectorstore, "get")):
                results = self.vectorstore.get(
                    where={"product_id": product_id}
                )
            # Fall back to older collection-based API if needed
            elif hasattr(self.vectorstore, "_collection"):
                results = self.vectorstore._collection.get(
                    where={"product_id": product_id}
                )
            else:
                raise ValueError("Could not find appropriate method to query vector store")
            
            # Return list of document IDs and their metadata
            return list(zip(results['ids'], results['metadatas']))
            
        except Exception as e:
            logger.error(f"Error identifying documents for product {product_id}: {str(e)}")
            logger.warning("Returning empty result set")
            return []
    
    def add_product_knowledge(
        self, 
        product_id: str, 
        document_paths: List[str],
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Add product knowledge to the vector database.
        
        Args:
            product_id: Unique identifier for the product
            document_paths: List of paths to documents containing product information
            metadata: Additional metadata to associate with all documents
            
        Returns:
            Number of document chunks added to the vector database
        """
        logger.info(f"Adding knowledge for product {product_id}")
        
        # Ensure base metadata exists
        base_metadata = metadata or {}
        base_metadata.update({
            "product_id": product_id,
            "status": "active",
            "added_date": datetime.now().isoformat()
        })
        
        # Load and process documents
        documents = []
        for doc_path in document_paths:
            # Choose loader based on file extension
            if doc_path.endswith('.pdf'):
                loader = PyPDFLoader(doc_path)
            elif doc_path.endswith('.txt'):
                loader = TextLoader(doc_path)
            elif os.path.isdir(doc_path):
                loader = DirectoryLoader(doc_path)
            else:
                logger.warning(f"Unsupported file type: {doc_path}, skipping")
                continue
            
            # Load the document
            logger.info(f"Loading document: {doc_path}")
            doc_documents = loader.load()
            
            # Add metadata to each document
            for doc in doc_documents:
                doc.metadata.update(base_metadata)
                # Add source file information
                doc.metadata["source"] = doc_path
                doc.metadata["filename"] = os.path.basename(doc_path)
            
            documents.extend(doc_documents)
        
        # Split documents into chunks
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add to vector store
        ids = self.vectorstore.add_documents(chunks)
        logger.info(f"Added {len(ids)} document chunks to vector database")
        
        # Update product relationship tracking
        if product_id not in self.product_relationships["products"]:
            self.product_relationships["products"][product_id] = {
                "status": "active",
                "added_date": base_metadata["added_date"],
                "document_count": len(chunks),
                "sources": [os.path.basename(path) for path in document_paths]
            }
            self._save_product_relationships()
        
        # Persist the vector store to disk if the method exists
        try:
            if hasattr(self.vectorstore, "persist") and callable(getattr(self.vectorstore, "persist")):
                self.vectorstore.persist()
            else:
                logger.info("No persist method found on vectorstore - newer Chroma versions persist automatically")
        except Exception as e:
            logger.warning(f"Error persisting vector store: {str(e)} - continuing anyway")
        
        return len(ids)
    
    def retire_product(
        self, 
        product_id: str, 
        replacement_product_id: Optional[str] = None,
        hard_delete: bool = False
    ) -> Dict[str, Any]:
        """
        Retire a product from the knowledge base.
        
        Args:
            product_id: ID of the product to retire
            replacement_product_id: ID of the product that replaces it (if any)
            hard_delete: Whether to completely remove the product (True) or just mark as retired (False)
            
        Returns:
            Dict containing operation results
        """
        logger.info(f"Retiring product {product_id}" + 
                   (f", replacement: {replacement_product_id}" if replacement_product_id else ""))
        
        # Find all documents for this product
        product_docs = self.identify_product_documents(product_id)
        
        if not product_docs:
            logger.warning(f"No documents found for product {product_id}")
            return {"status": "warning", "message": f"No documents found for product {product_id}"}
        
        # Track the operation in product relationships
        retirement_date = datetime.now().isoformat()
        
        # Update product status in relationship tracking
        if product_id in self.product_relationships["products"]:
            self.product_relationships["products"][product_id].update({
                "status": "retired",
                "retirement_date": retirement_date
            })
        
        # Record replacement relationship if provided
        if replacement_product_id:
            self.product_relationships["replacements"][product_id] = replacement_product_id
            
            # Add to history
            self.product_relationships["history"].append({
                "date": retirement_date,
                "action": "retirement",
                "product_id": product_id,
                "replacement_product_id": replacement_product_id
            })
        
        # Save relationship changes
        self._save_product_relationships()
        
        if hard_delete:
            # Hard delete: remove documents from vector store
            doc_ids = [doc_id for doc_id, _ in product_docs]
            self.vectorstore.delete(ids=doc_ids)
            logger.info(f"Hard deleted {len(doc_ids)} documents for product {product_id}")
            result = {
                "status": "success", 
                "action": "hard_delete",
                "deleted_documents": len(doc_ids)
            }
        else:
            # Soft delete: update metadata to mark as retired
            doc_ids = [doc_id for doc_id, _ in product_docs]
            
            # Update metadata for each document
            updated_count = 0
            for doc_id, metadata in product_docs:
                metadata.update({
                    "status": "retired",
                    "retirement_date": retirement_date
                })
                
                if replacement_product_id:
                    metadata["replacement_product_id"] = replacement_product_id
                
                # Update in Chroma - need to use lower-level API to update metadata only
                self.vectorstore._collection.update(
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                updated_count += 1
            
            logger.info(f"Soft retired {updated_count} documents for product {product_id}")
            result = {
                "status": "success", 
                "action": "soft_retire",
                "updated_documents": updated_count
            }
        
        # Persist changes if the method exists
        try:
            if hasattr(self.vectorstore, "persist") and callable(getattr(self.vectorstore, "persist")):
                self.vectorstore.persist()
            else:
                logger.info("No persist method found on vectorstore - newer Chroma versions persist automatically")
        except Exception as e:
            logger.warning(f"Error persisting vector store: {str(e)} - continuing anyway")
        
        return result

    def replace_product(
        self,
        old_product_id: str,
        new_product_id: str,
        new_product_docs: List[str],
        new_product_metadata: Dict[str, Any] = None,
        hard_delete: bool = False
    ) -> Dict[str, Any]:
        """
        Complete workflow to replace an old product with a new one.
        
        Args:
            old_product_id: ID of the product being replaced
            new_product_id: ID of the new replacement product
            new_product_docs: Paths to documents for the new product
            new_product_metadata: Additional metadata for the new product
            hard_delete: Whether to completely remove old product info
            
        Returns:
            Dict containing operation results
        """
        logger.info(f"Starting product replacement: {old_product_id} → {new_product_id}")
        
        # Step 1: Retire the old product
        retire_result = self.retire_product(
            product_id=old_product_id,
            replacement_product_id=new_product_id,
            hard_delete=hard_delete
        )
        
        # Step 2: Add knowledge for the new product
        # Add relationship info to metadata
        new_metadata = new_product_metadata or {}
        new_metadata.update({
            "replaces_product_id": old_product_id
        })
        
        # Add the new product documents
        added_chunks = self.add_product_knowledge(
            product_id=new_product_id,
            document_paths=new_product_docs,
            metadata=new_metadata
        )
        
        # Step 3: Update any cross-references in other documents
        updated_refs = self._update_product_references(
            old_product_id=old_product_id,
            new_product_id=new_product_id
        )
        
        # Record the complete replacement operation
        result = {
            "status": "success",
            "old_product": old_product_id,
            "new_product": new_product_id,
            "retirement": retire_result,
            "new_documents_added": added_chunks,
            "cross_references_updated": updated_refs
        }
        
        logger.info(f"Product replacement complete: {old_product_id} → {new_product_id}")
        return result
    
    def _update_product_references(
        self,
        old_product_id: str,
        new_product_id: str
    ) -> int:
        """
        Find and update references to the old product in other product documents.
        
        Args:
            old_product_id: ID of the old product
            new_product_id: ID of the new product
            
        Returns:
            Number of documents updated
        """
        logger.info(f"Updating cross-references from {old_product_id} to {new_product_id}")
        
        # Find documents that mention the old product but aren't about it directly
        query = f"compatible with {old_product_id} or works with {old_product_id}"
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=50  # Get up to 50 potentially relevant documents
        )
        
        # Filter to only get documents that aren't about the old product directly
        cross_ref_docs = []
        for doc, _ in results:
            if doc.metadata.get("product_id") != old_product_id:
                cross_ref_docs.append(doc)
        
        logger.info(f"Found {len(cross_ref_docs)} documents with potential cross-references")
        
        # We would typically need NLP here to accurately identify and update references
        # For this example, we'll use a simplified approach to illustrate the concept
        updated_count = 0
        
        # For each document with potential references
        for doc in cross_ref_docs:
            # Check if it explicitly mentions the old product ID
            if old_product_id in doc.page_content:
                doc_id = doc.metadata.get("doc_id")  # Assuming Chroma assigns this
                
                # Simple text replacement - in practice you'd want more sophisticated NLP
                updated_content = doc.page_content.replace(
                    old_product_id, 
                    f"{new_product_id} (which replaces {old_product_id})"
                )
                
                # Update the document's metadata to mark it as updated
                doc.metadata["updated_date"] = datetime.now().isoformat()
                doc.metadata["references_updated"] = True
                doc.metadata["references"] = doc.metadata.get("references", []) + [new_product_id]
                
                # In practice, we would need to:
                # 1. Remove the old document
                # 2. Re-embed the updated content
                # 3. Add the new embedding with the updated metadata
                
                # Simplified example only - in reality you'd need to delete and re-add
                # since we can't update the content/embedding directly in most vector stores
                logger.info(f"Would update cross-reference in document {doc_id}")
                updated_count += 1
        
        logger.info(f"Identified {updated_count} documents for cross-reference updates")
        return updated_count


class ProductAwareRAG:
    """
    RAG implementation that is aware of product lifecycles and handles
    queries about retired products appropriately.
    """
    
    def __init__(
        self,
        knowledge_manager: ProductKnowledgeManager,
        llm: Optional[BaseChatModel] = None
    ):
        """
        Initialize the product-aware RAG system.
        
        Args:
            knowledge_manager: The product knowledge manager
            llm: The LLM to use for generation (defaults to using create_llm() to select the best available provider)
        """
        self.knowledge_manager = knowledge_manager
        
        # Use the provided LLM or create one using our factory function
        self.llm = llm if llm is not None else create_llm()
        
        # Get the LLM name for display purposes
        if hasattr(self.llm, 'name'):
            self.llm_name = self.llm.name
        elif hasattr(self.llm, 'model_name'):
            self.llm_name = f"OpenAI ({self.llm.model_name})"
        else:
            self.llm_name = self.llm._llm_type
            
        logger.info(f"ProductAwareRAG initialized with LLM: {self.llm_name}")
        
        # Initialize prompt templates
        self._init_prompts()
        
        # Build the RAG chain
        self._build_rag_chain()
    
    def _init_prompts(self) -> None:
        """Initialize prompt templates for the RAG system."""
        # Base RAG prompt
        self.rag_prompt = ChatPromptTemplate.from_template("""
        You are a product information assistant. Answer the question based on the context provided.
        
        Context information is below:
        ---------------------
        {context}
        ---------------------
        
        Given the context information and no prior knowledge, answer the following question:
        {question}
        """)
        
        # Product-aware prompt that handles retired products
        self.product_aware_prompt = ChatPromptTemplate.from_template("""
        You are a product information assistant. Answer the question based on the context provided.
        
        Context information is below:
        ---------------------
        {context}
        ---------------------
        
        IMPORTANT PRODUCT STATUS INFORMATION:
        {product_status}
        
        Given the context information and the product status information, answer the following question:
        {question}
        
        If the question is about a retired product, acknowledge this and direct the user to the replacement product.
        """)
    
    def _build_rag_chain(self) -> None:
        """Build the core RAG chain for answering product queries."""
        # Create a retriever from the vector store
        self.retriever = self.knowledge_manager.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Basic RAG chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Product-aware RAG chain with status checking
        self.product_aware_chain = (
            {
                "context": self.retriever, 
                "question": RunnablePassthrough(),
                "product_status": lambda x: self._get_product_status(x)
            }
            | self.product_aware_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _get_product_status(self, query: str) -> str:
        """
        Extract product mentions from the query and get their status information.
        
        Args:
            query: The user's question
            
        Returns:
            String containing product status information relevant to the query
        """
        # In a real system, you would use NLP to extract product mentions
        # For this example, we'll use a simplified approach with the product relationships
        
        status_info = []
        for product_id in self.knowledge_manager.product_relationships["products"]:
            # Simple check if product is mentioned in query
            if product_id.lower() in query.lower():
                product_info = self.knowledge_manager.product_relationships["products"][product_id]
                
                if product_info["status"] == "retired":
                    # Check if there's a replacement
                    replacement_id = self.knowledge_manager.product_relationships["replacements"].get(product_id)
                    
                    if replacement_id:
                        status_info.append(
                            f"Product {product_id} has been retired and replaced by {replacement_id}."
                        )
                    else:
                        status_info.append(f"Product {product_id} has been retired.")
        
        if not status_info:
            return "All products mentioned are currently active."
        
        return "\n".join(status_info)
    
    def query(
        self, 
        question: str,
        include_retired: bool = True,
        use_product_awareness: bool = True
    ) -> str:
        """
        Query the RAG system with product awareness.
        
        Args:
            question: The user's question
            include_retired: Whether to include retired products in search results
            use_product_awareness: Whether to use product status awareness
            
        Returns:
            The generated answer
        """
        logger.info(f"Processing query: {question}")
        
        # Determine which search filter to use based on include_retired flag
        if not include_retired:
            # Set up the retriever to exclude retired products
            self.retriever.search_kwargs["filter"] = {"status": "active"}
        else:
            # Clear any filters to include all documents
            self.retriever.search_kwargs.pop("filter", None)
        
        # Use the appropriate chain based on product awareness setting
        if use_product_awareness:
            return self.product_aware_chain.invoke(question)
        else:
            return self.chain.invoke(question)


def run_product_replacement_demo(vector_db_path="./demo_vector_db"):
    """
    Run a demonstration of the product replacement workflow.
    
    Args:
        vector_db_path: Path to the vector database directory
    """
    print("RAG Product Replacement Workflow Demo")
    print("=====================================")
    print(f"Using vector database path: {vector_db_path}")
    
    # Initialize the knowledge manager
    # Note: This demo will use mock embeddings if OpenAI API key is not available
    manager = ProductKnowledgeManager(vector_db_path=vector_db_path)
    
    # Step 1: Add knowledge for the original product
    print("\n1. Adding knowledge for the original product (ProductA)")
    
    # Create some sample product documents
    os.makedirs("./demo_docs", exist_ok=True)
    
    with open("./demo_docs/productA_specs.txt", "w") as f:
        f.write("""
        # ProductA Specifications
        
        Model: ProductA-2023
        Processing power: 2.4 GHz
        Memory: 8GB
        Storage: 256GB SSD
        Battery life: 10 hours
        Weight: 1.8 kg
        
        ProductA is compatible with all standard accessories and can be upgraded
        to the latest software version.
        """)
    
    with open("./demo_docs/productA_manual.txt", "w") as f:
        f.write("""
        # ProductA User Manual
        
        Thank you for purchasing ProductA!
        
        ## Setup Instructions
        
        1. Unbox the ProductA
        2. Connect to power
        3. Press the power button for 3 seconds
        4. Follow on-screen instructions
        
        ## Troubleshooting
        
        If ProductA does not power on, check the power connection and ensure
        the battery is charged.
        """)
    
    # Add product knowledge
    manager.add_product_knowledge(
        product_id="ProductA",
        document_paths=["./demo_docs/productA_specs.txt", "./demo_docs/productA_manual.txt"],
        metadata={"version": "2023", "category": "electronic"}
    )
    
    # Step 2: Add knowledge for a complementary product that references ProductA
    print("\n2. Adding knowledge for a complementary product that references ProductA")
    
    with open("./demo_docs/accessory_specs.txt", "w") as f:
        f.write("""
        # ProductAccessory Specifications
        
        This accessory is designed to work with ProductA.
        
        It enhances the functionality of ProductA by adding additional ports
        and extending battery life by up to 5 hours.
        
        Only compatible with ProductA and similar models.
        """)
    
    manager.add_product_knowledge(
        product_id="ProductAccessory",
        document_paths=["./demo_docs/accessory_specs.txt"],
        metadata={"version": "1.0", "category": "accessory"}
    )
    
    # Step 3: Create the replacement product documentation
    print("\n3. Creating documentation for the replacement product (ProductB)")
    
    with open("./demo_docs/productB_specs.txt", "w") as f:
        f.write("""
        # ProductB Specifications
        
        Model: ProductB-2025
        Processing power: 3.6 GHz
        Memory: 16GB
        Storage: 512GB SSD
        Battery life: 15 hours
        Weight: 1.5 kg
        
        ProductB is the next generation replacement for ProductA, with improved
        performance and battery life. It maintains compatibility with most
        ProductA accessories.
        """)
    
    with open("./demo_docs/productB_manual.txt", "w") as f:
        f.write("""
        # ProductB User Manual
        
        Thank you for purchasing ProductB!
        
        ## Setup Instructions
        
        1. Unbox the ProductB
        2. Connect to power
        3. Press the power button for 2 seconds
        4. Follow on-screen instructions
        
        ## Migration from ProductA
        
        If you're upgrading from ProductA, you can transfer your data using
        the ProductB Migration Assistant.
        
        ## Troubleshooting
        
        If ProductB does not power on, check the power connection and ensure
        the battery is charged.
        """)
    
    # Step 4: Execute the product replacement workflow
    print("\n4. Executing the product replacement workflow")
    
    result = manager.replace_product(
        old_product_id="ProductA",
        new_product_id="ProductB",
        new_product_docs=["./demo_docs/productB_specs.txt", "./demo_docs/productB_manual.txt"],
        new_product_metadata={"version": "2025", "category": "electronic"},
        hard_delete=False  # Use soft deletion to maintain historical context
    )
    
    print(f"Replacement result: {json.dumps(result, indent=2)}")
    
    # Step 5: Set up the RAG system and run some queries
    print("\n5. Setting up the RAG system and running queries")
    
    rag = ProductAwareRAG(manager)
    print(f"\nUsing {rag.llm_name} for text generation")
    
    # Test query about the retired product
    print("\nQuery about the retired product:")
    query1 = "What are the specifications of ProductA?"
    print(f"Q: {query1}")
    answer1 = rag.query(query1)
    print(f"A: {answer1}")
    
    # Test query about the new product
    print("\nQuery about the new product:")
    query2 = "What are the specifications of ProductB?"
    print(f"Q: {query2}")
    answer2 = rag.query(query2)
    print(f"A: {answer2}")
    
    # Test query about compatibility
    print("\nQuery about compatibility:")
    query3 = "Is ProductAccessory compatible with the latest products?"
    print(f"Q: {query3}")
    answer3 = rag.query(query3)
    print(f"A: {answer3}")
    
    # Test query filtering out retired products
    print("\nQuery with retired products filtered out:")
    query4 = "What are the specifications of ProductA?"
    print(f"Q: {query4}")
    answer4 = rag.query(query4, include_retired=False)
    print(f"A: {answer4}")
    
    print("\nDemo complete!")


def print_debug_info():
    """Print debugging information to help diagnose issues."""
    print("\n=== Debug Information ===")
    
    # Check OpenAI API key
    if "OPENAI_API_KEY" in os.environ:
        key = os.environ["OPENAI_API_KEY"]
        # Only show the first 5 and last 5 characters of the key
        if len(key) > 12:
            masked_key = f"{key[:5]}...{key[-5:]}"
        else:
            masked_key = "***SET BUT TOO SHORT***"
        print(f"OPENAI_API_KEY: {masked_key} (length: {len(key)})")
    else:
        print("OPENAI_API_KEY=unset")
    
    # Check Ollama availability
    try:
        response = requests.get(f"{LLMConfig.OLLAMA_BASE_URL}/api/tags", timeout=2)
        print(f"Ollama Status: Available (HTTP {response.status_code})")
        try:
            models = response.json().get("models", [])
            if models:
                model_names = [model.get("name") for model in models]
                print(f"Ollama Models: {', '.join(model_names)}")
            else:
                print("Ollama Models: None found")
        except Exception:
            print("Ollama Models: Error parsing response")
    except Exception as e:
        print(f"Ollama Status: Not available ({str(e)})")
    
    print("========================")

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="RAG Product Replacement Demo")
    parser.add_argument("--vector-db", "-v", default="./demo_vector_db",
                      help="Path to vector database (default: ./demo_vector_db)")
    parser.add_argument("--clean", "-c", action="store_true",
                      help="Clean existing vector database before running")
    parser.add_argument("--debug", "-d", action="store_true",
                      help="Enable additional debug output")
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set up more verbose logging if debug flag is set
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug logging enabled")
        
        # Clean vector database if requested
        if args.clean:
            import shutil
            if os.path.exists(args.vector_db):
                print(f"Cleaning vector database at {args.vector_db}")
                shutil.rmtree(args.vector_db, ignore_errors=True)
                os.makedirs(args.vector_db, exist_ok=True)
        
        # Print debug info at the start
        print_debug_info()
        
        # Run the demo with the specified vector database path
        run_product_replacement_demo(vector_db_path=args.vector_db)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        
        # Print debug info again on error
        print_debug_info()
        
        error_msg = str(e).lower()
        if "quota" in error_msg or "insufficient_quota" in error_msg or "429" in error_msg:
            print("\nERROR: OpenAI API quota exceeded or rate limit hit.")
            print("Please check your billing details at https://platform.openai.com/account/billing")
            print("Alternatively, you can use Ollama instead.")
        elif "dimension" in error_msg and "does not match" in error_msg:
            print("\nERROR: Embedding dimension mismatch in the vector database.")
            print(f"Details: {str(e)}")
            print("\nPossible solutions:")
            print("1. Run with --clean flag to start with a fresh database: python rag-update-demo.py --clean")
            print("2. Use a different directory: python rag-update-demo.py --vector-db ./new_vector_db")
            print("3. Manually delete the database: rm -rf ./demo_vector_db")
        elif "openai" in error_msg and "api" in error_msg:
            print("\nERROR: There was an issue with the OpenAI API.")
            print(f"Details: {str(e)}")
            print("Alternatively, you can use Ollama instead.")
        elif "chroma" in error_msg or "vector" in error_msg or "database" in error_msg:
            print("\nERROR: There was an issue with the vector database.")
            print(f"Details: {str(e)}")
            print("\nPossible solutions:")
            print("1. Run with --clean flag to start with a fresh database: python rag-update-demo.py --clean")
            print("2. Use a different directory: python rag-update-demo.py --vector-db ./new_vector_db")
            print("3. Manually delete the database: rm -rf ./demo_vector_db")
        else:
            print(f"\nERROR: {str(e)}")
            print("\nTroubleshooting:")
            print("- Both OpenAI API key and Ollama server appear to be available")
            print("- Check the error message above for more specific details")
            print("- Try running with --clean flag: python rag-update-demo.py --clean")
            print("- Try running with --debug flag for more verbose output: python rag-update-demo.py --debug")
        
        exit(1)
    