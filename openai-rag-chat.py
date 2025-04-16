"""
LLM-Powered RAG Chat Interface

This script provides a text-based chat interface that:
1. Uses either OpenAI API or local Ollama for response generation
2. Reads from the same JSON knowledge base as the RAG system
3. Monitors for changes to the product information in real-time
4. Clearly indicates which LLM is being used (OpenAI or Ollama)

Note: This script only provides the chat interface. To update product information, 
use the rag-update-demo.py script.

Requirements:
- openai (pip install openai) for OpenAI API
- requests (pip install requests) for Ollama API

Usage:
python openai-rag-chat.py [options]

Options:
  --vector-db PATH, -v PATH   Specify custom vector database path (default: ./demo_vector_db)
  --debug, -d                 Enable additional debug output
"""

import os
import json
import time
import threading
import cmd
import requests
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    import openai
    has_openai = True
except ImportError:
    has_openai = False


class LLMConfig:
    """Configuration for LLM providers"""
    
    # Ollama configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2"  # Default model
    
    # OpenAI configuration
    OPENAI_MODEL = "gpt-3.5-turbo"


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self):
        self.name = "Base LLM"
        self.model = "Unknown"
    
    def validate(self):
        """Validate the LLM provider configuration"""
        return False
    
    def generate_response(self, prompt, system_message="You are a helpful assistant"):
        """Generate a response using the LLM"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_display_name(self):
        """Get a display name for the prompt"""
        return f"{self.name} - {self.model}"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self):
        super().__init__()
        self.name = "OpenAI"
        self.client = None
        self.model = LLMConfig.OPENAI_MODEL
    
    def validate(self):
        """Validate OpenAI API key and connection"""
        if not has_openai:
            logger.warning("OpenAI package not installed.")
            return False
            
        if "OPENAI_API_KEY" not in os.environ:
            logger.warning("OPENAI_API_KEY environment variable not set.")
            return False
            
        try:
            # Initialize OpenAI client
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self.client = openai.OpenAI(api_key=openai.api_key)
            
            # Make a small test call to validate the API key
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "This is a test."},
                    {"role": "user", "content": "Say 'API key is valid' if you receive this."}
                ],
                max_tokens=20,
                temperature=0
            )
            
            # Check response content
            response_text = test_response.choices[0].message.content.strip().lower()
            if "valid" not in response_text and "api key" not in response_text:
                logger.warning(f"OpenAI returned unexpected response: {response_text}")
                
            logger.info("OpenAI API key validated successfully.")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "insufficient_quota" in error_msg or "429" in error_msg:
                logger.error("OpenAI API quota exceeded or rate limit hit. Please check your billing details.")
            else:
                logger.warning(f"Error validating OpenAI API key: {str(e)}")
            return False
    
    def generate_response(self, prompt, system_message="You are a helpful assistant"):
        """Generate a response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "insufficient_quota" in error_msg or "429" in error_msg:
                raise Exception(f"OpenAI API quota exceeded. Check your billing details at https://platform.openai.com/account/billing. Error: {str(e)}")
            else:
                raise Exception(f"OpenAI API error: {str(e)}")


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self):
        super().__init__()
        self.name = "Ollama"
        self.base_url = LLMConfig.OLLAMA_BASE_URL
        self.model = LLMConfig.OLLAMA_MODEL
    
    def validate(self):
        """Validate Ollama server connection"""
        try:
            # Check if Ollama server is running and model is available
            logger.info(f"Checking Ollama server at {self.base_url}")
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
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
            logger.warning("Make sure Ollama is running locally on port 11434")
            return False
    
    def generate_response(self, prompt, system_message="You are a helpful assistant"):
        """Generate a response using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_message,
                "stream": False
            }
            
            logger.debug(f"Sending request to Ollama with model: {self.model}")
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"Ollama server returned status code {response.status_code}")
                
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise Exception(f"Ollama API error: {str(e)}")


class LLMFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_llm():
        """Create and return an LLM provider based on availability"""
        logger.info("Initializing LLM provider")
        
        # Try OpenAI first
        logger.info("Checking OpenAI availability...")
        openai_provider = OpenAIProvider()
        if openai_provider.validate():
            return openai_provider
        
        # Then try Ollama
        logger.info("Checking Ollama availability...")
        ollama_provider = OllamaProvider()
        if ollama_provider.validate():
            return ollama_provider
            
        # If both fail, raise an exception
        error_message = """
No LLM provider available. Please configure one of the following:
1. Set OPENAI_API_KEY environment variable with a valid API key
2. Start a local Ollama server (http://localhost:11434) with at least one model installed
"""
        logger.error(error_message)
        raise Exception(error_message)


class ProductChatAssistant(cmd.Cmd):
    """
    Interactive chat interface for querying product information using an LLM.
    """
    
    intro = """
==============================================
    Product Sales Assistant
==============================================
Welcome! I'm your virtual sales assistant.
Ask me anything about our products, features, specifications, 
compatibility, or how to set them up.

Type 'help' for commands, 'exit' to quit.
"""
    
    def __init__(self, db_path: str = "./demo_vector_db"):
        """Initialize the chat interface."""
        super().__init__()
        
        self.db_path = db_path
        self.relationships_file = os.path.join(db_path, "product_relationships.json")
        logger.info(f"Using vector database path: {db_path}")
        print(f"Using vector database path: {db_path}")
        
        # Initialize LLM provider
        try:
            self.llm = LLMFactory.create_llm()
            logger.info(f"Using {self.llm.name} with model {self.llm.model} for response generation")
            print(f"Using {self.llm.name} with model {self.llm.model} for response generation")
            self.prompt = f"\n[{self.llm.get_display_name()}] You: "
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            print(f"Error initializing LLM: {str(e)}")
            print("Exiting...")
            raise Exception(f"Failed to initialize LLM provider: {str(e)}")
        
        # Load product info
        self._load_product_info()
        
        # Product details for context
        self.product_details = {
            "ProductA": {
                "specs": "2.4 GHz processing power, 8GB memory, 256GB SSD, 10-hour battery life, 1.8 kg",
                "setup": "1. Unbox the ProductA\n2. Connect to power\n3. Press the power button for 3 seconds\n4. Follow on-screen instructions",
                "troubleshooting": "If ProductA does not power on, check the power connection and ensure the battery is charged."
            },
            "ProductB": {
                "specs": "3.6 GHz processing power, 16GB memory, 512GB SSD, 15-hour battery life, 1.5 kg",
                "setup": "1. Unbox the ProductB\n2. Connect to power\n3. Press the power button for 2 seconds\n4. Follow on-screen instructions",
                "troubleshooting": "If ProductB does not power on, check the power connection and ensure the battery is charged.",
                "migration": "If you're upgrading from ProductA, you can transfer your data using the ProductB Migration Assistant."
            },
            "ProductAccessory": {
                "specs": "Enhances functionality by adding additional ports and extending battery life by up to 5 hours.",
                "compatibility": {
                    "ProductA": "Fully compatible with ProductA.",
                    "ProductB": "Limited compatibility with ProductB."
                }
            }
        }
        
        # Start a background thread to monitor database changes
        self.last_check_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_db_changes)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _load_product_info(self):
        """Load product information from the file."""
        if os.path.exists(self.relationships_file):
            logger.info(f"Loading product information from {self.relationships_file}")
            try:
                with open(self.relationships_file, 'r') as f:
                    self.product_info = json.load(f)
                product_count = len(self.product_info.get('products', {}))
                logger.info(f"Loaded information for {product_count} products")
                print(f"Loaded information for {product_count} products")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing product information file: {str(e)}")
                print(f"Error parsing product information file: {str(e)}")
                self.product_info = {"products": {}, "replacements": {}, "history": []}
        else:
            logger.warning(f"Product information file not found: {self.relationships_file}")
            self.product_info = {"products": {}, "replacements": {}, "history": []}
            print("No product information found")
    
    def _monitor_db_changes(self):
        """Background thread to monitor changes in the database."""
        while self.monitoring:
            time.sleep(2)  # Check every 2 seconds
            
            # Check if the product relationships file has been modified
            if os.path.exists(self.relationships_file):
                modified_time = os.path.getmtime(self.relationships_file)
                
                if modified_time > self.last_check_time:
                    # Reload the product info
                    self._load_product_info()
                    
                    # Use the LLM name in the prefix
                    prefix = f"[{self.llm.get_display_name()}]"
                    print(f"\n\n{prefix} [System] Product information has been updated!")
                    
                    # Print a summary of changes
                    retired_products = [p for p, info in self.product_info["products"].items() 
                                      if info.get("status") == "retired"]
                    
                    if retired_products:
                        print(f"{prefix} [System] Retired products: {', '.join(retired_products)}")
                        for product in retired_products:
                            replacement = self.product_info["replacements"].get(product)
                            if replacement:
                                print(f"{prefix} [System] {product} has been replaced by {replacement}")
                    
                    active_products = [p for p, info in self.product_info["products"].items() 
                                     if info.get("status") == "active"]
                    if active_products:
                        print(f"{prefix} [System] Active products: {', '.join(active_products)}")
                    
                    print(f"\n{prefix} You: ", end="", flush=True)
                    
                    self.last_check_time = time.time()
    
    def _get_product_status(self, product_id):
        """Get the status of a product."""
        if product_id in self.product_info.get("products", {}):
            return self.product_info["products"][product_id].get("status")
        return None
    
    def _get_replacement(self, product_id):
        """Get replacement for a retired product."""
        return self.product_info.get("replacements", {}).get(product_id)
    
    def _generate_context_string(self):
        """Generate a context string with all product information."""
        context = []
        
        # Add product status information
        for product_id, details in self.product_info.get("products", {}).items():
            status = details.get("status", "unknown")
            if status == "retired":
                replacement = self._get_replacement(product_id)
                if replacement:
                    context.append(f"{product_id} has been retired and replaced by {replacement}.")
                else:
                    context.append(f"{product_id} has been retired.")
            else:
                context.append(f"{product_id} is an active product.")
        
        # Add specific product details
        for product_id, details in self.product_details.items():
            context.append(f"\n{product_id} Specifications:")
            if "specs" in details:
                context.append(details["specs"])
            
            # Add compatibility info for accessories
            if product_id == "ProductAccessory" and "compatibility" in details:
                context.append(f"\n{product_id} Compatibility:")
                for compat_product, compat_info in details["compatibility"].items():
                    context.append(f"- {compat_info}")
        
        return "\n".join(context)
    
    def default(self, line):
        """Process user input as a question to the system."""
        if line.lower() in ('exit', 'quit', 'bye'):
            return self.do_exit()
        
        try:
            # Generate a response using the LLM
            context = self._generate_context_string()
            
            # Create the prompt
            prompt = f"""Answer the following question based only on the context provided.

Context information:
{context}

User question: {line}

Important: 
- If a product is retired, make sure to mention this and direct the user to its replacement.
- Only use information from the context. Do not make up information.
- Keep your answer concise and focused on the question.
"""
            
            # Generate response with an enhanced system message
            system_message = """You are a helpful, enthusiastic sales assistant providing product information.

As a product information specialist:
- Format your responses for easy readability with appropriate spacing and structure
- Use bullet points for lists of features or specifications
- Highlight key selling points with bold formatting using markdown (**key point**)
- Organize information into clear sections when appropriate
- Be conversational, friendly, and professional
- When addressing product replacements, express enthusiasm about the newer product

Remember to only use information from the context provided. Do not make up information."""
            
            answer = self.llm.generate_response(prompt, system_message)
            
            print(f"\nAssistant: {answer}")
            
        except Exception as e:
            print(f"\nAssistant: I'm sorry, I encountered an error: {str(e)}")
    
    def do_exit(self, arg=None):
        """Exit the chat interface."""
        print("Goodbye!")
        self.monitoring = False
        return True
    
    def do_status(self, arg):
        """Show the current status of product information."""
        # Use the LLM name in the prefix
        prefix = f"[{self.llm.get_display_name()}]"
        
        print(f"\n{prefix} Current Product Availability:")
        print("=============================")
        
        if not self.product_info.get("products"):
            print("No product information available at this time.")
            print("Please check back later for our exciting product lineup!")
            return
        
        active_products = []
        retired_products = []
        
        for product, info in self.product_info["products"].items():
            status = info.get("status", "unknown")
            if status == "active":
                active_products.append(product)
            elif status == "retired":
                retired_products.append((product, self.product_info["replacements"].get(product)))
        
        if active_products:
            print("Currently Available Products:")
            for product in active_products:
                print(f"✓ {product}")
            
        if retired_products:
            print("\nRetired Products:")
            for product, replacement in retired_products:
                if replacement:
                    print(f"✗ {product} - Upgraded to our new and improved {replacement}!")
                else:
                    print(f"✗ {product} - No longer available")
    
    def do_help(self, arg):
        """Display help information."""
        # Use the LLM name in the prefix
        prefix = f"[{self.llm.get_display_name()}]"
        
        print(f"\n{prefix} Available commands:")
        print("  status    - Show current product availability")
        print("  exit/quit - End the conversation")
        print("\nSample questions you can ask:")
        print("  What are the key features of ProductB?")
        print("  How does ProductB compare to ProductA?")
        print("  Is ProductAccessory compatible with ProductB?")
        print("  What makes ProductB better than its predecessor?")
        print("  How do I set up my new ProductB?")
        print("  What products would you recommend for my needs?")
        print("  Which products are currently available?")


def run_chat_interface(db_path="./demo_vector_db"):
    """
    Run the chat interface.
    
    Args:
        db_path: Path to the vector database directory
    """
    try:
        chat = ProductChatAssistant(db_path=db_path)
        chat.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your environment setup and try again.")


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
    parser = argparse.ArgumentParser(description="RAG Chat Interface")
    parser.add_argument("--vector-db", "-v", default="./demo_vector_db",
                      help="Path to vector database (default: ./demo_vector_db)")
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
            logger.debug("Debug logging enabled")
        
        # Print debug info at the start
        print_debug_info()
        
        # Check if the database path exists
        if not os.path.exists(args.vector_db):
            logger.error(f"Database path not found: {args.vector_db}")
            print(f"Database path not found: {args.vector_db}")
            print("Please run the product replacement demo first: python rag-update-demo.py")
            exit(1)
        
        logger.info(f"Starting chat interface with database at {args.vector_db}")
        print(f"Starting chat interface with database at {args.vector_db}")
        print("Note: To update product information, run the separate script: python rag-update-demo.py")
        
        # Run the chat interface
        run_chat_interface(db_path=args.vector_db)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        
        # Print debug info again on error
        print_debug_info()
        
        error_msg = str(e).lower()
        if "openai" in error_msg and "api" in error_msg:
            if "quota" in error_msg or "insufficient_quota" in error_msg or "429" in error_msg:
                print("\nERROR: OpenAI API quota exceeded or rate limit hit.")
                print("Please check your billing details at https://platform.openai.com/account/billing")
                print("Alternatively, you can use Ollama instead.")
            else:
                print("\nERROR: There was an issue with the OpenAI API.")
                print(f"Details: {str(e)}")
                print("Alternatively, you can use Ollama instead.")
        elif "ollama" in error_msg:
            print("\nERROR: There was an issue connecting to Ollama server.")
            print(f"Details: {str(e)}")
            print("Make sure Ollama is running locally on port 11434")
        else:
            print(f"\nERROR: {str(e)}")
            print("\nTroubleshooting:")
            print("- Try running with --debug flag for more verbose output: python openai-rag-chat.py --debug")
            print("- Make sure the demo_vector_db directory exists by running: python rag-update-demo.py")
            print("- Ensure either OpenAI API key is set or Ollama server is running")
        
        exit(1)