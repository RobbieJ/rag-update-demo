"""
LLM-Powered RAG Chat Interface with Live Updates

This script provides a text-based chat interface that:
1. Uses either OpenAI API or local Ollama for response generation
2. Reads from the same JSON knowledge base as the RAG system
3. Monitors for changes to the product information
4. Clearly indicates which LLM is being used (OpenAI or Ollama)

Requirements:
- openai (pip install openai) for OpenAI API
- requests (pip install requests) for Ollama API
"""

import os
import json
import time
import threading
import cmd
import random
import requests
from datetime import datetime

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
    
    def validate(self):
        """Validate the LLM provider configuration"""
        return False
    
    def generate_response(self, prompt, system_message="You are a helpful assistant"):
        """Generate a response using the LLM"""
        raise NotImplementedError("Subclasses must implement this method")


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
            print("OpenAI package not installed.")
            return False
            
        if "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY environment variable not set.")
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
            
            print("OpenAI API key validated successfully.")
            return True
            
        except Exception as e:
            print(f"Error validating OpenAI API key: {str(e)}")
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
            response = requests.get(f"{self.base_url}/api/tags")
            
            if response.status_code != 200:
                print(f"Ollama server returned status code {response.status_code}")
                return False
                
            # Check if our model is available
            available_models = response.json().get("models", [])
            model_names = [model.get("name") for model in available_models]
            
            if not model_names:
                print("No models found on Ollama server")
                return False
                
            if self.model not in model_names:
                print(f"Model {self.model} not found. Available models: {', '.join(model_names)}")
                print(f"Using {model_names[0]} instead.")
                self.model = model_names[0]
            
            print(f"Ollama server validated successfully with model: {self.model}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama server: {str(e)}")
            print("Make sure Ollama is running locally on port 11434")
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
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Ollama server returned status code {response.status_code}")
                
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")


class LLMFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_llm():
        """Create and return an LLM provider based on availability"""
        # Try OpenAI first
        openai_provider = OpenAIProvider()
        if openai_provider.validate():
            return openai_provider
            
        # Then try Ollama
        ollama_provider = OllamaProvider()
        if ollama_provider.validate():
            return ollama_provider
            
        # If both fail, raise an exception
        raise Exception("No LLM provider available. Please configure either OpenAI API key or start Ollama server.")


class ProductChatAssistant(cmd.Cmd):
    """
    Interactive chat interface for querying product information using an LLM.
    """
    
    intro = """
==============================================
    Product Information Chat Assistant
==============================================
Ask questions about our products to get information.
Type 'help' for commands, 'exit' to quit.
"""
    
    def __init__(self, db_path: str = "./demo_vector_db"):
        """Initialize the chat interface."""
        super().__init__()
        
        self.db_path = db_path
        self.relationships_file = os.path.join(db_path, "product_relationships.json")
        
        # Initialize LLM provider
        try:
            self.llm = LLMFactory.create_llm()
            print(f"Using {self.llm.name} for response generation")
            self.prompt = f"\n[{self.llm.name}] You: "
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            print("Exiting...")
            exit(1)
        
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
            with open(self.relationships_file, 'r') as f:
                self.product_info = json.load(f)
            print(f"Loaded information for {len(self.product_info.get('products', {}))} products")
        else:
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
                    prefix = f"[{self.llm.name}]"
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
            prompt = f"""You are a product information assistant. Answer the following question based only on the context provided.

Context information:
{context}

User question: {line}

Important: 
- If a product is retired, make sure to mention this and direct the user to its replacement.
- Only use information from the context. Do not make up information.
- Keep your answer concise and focused on the question.
"""
            
            # Generate response
            system_message = "You are a product information assistant."
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
        prefix = f"[{self.llm.name}]"
        
        print(f"\n{prefix} Current Product Information:")
        print("===========================")
        
        if not self.product_info.get("products"):
            print("No product information available")
            return
        
        for product, info in self.product_info["products"].items():
            status = info.get("status", "unknown")
            status_str = "ACTIVE" if status == "active" else "RETIRED"
            print(f"{product}: {status_str}")
            
            if status == "retired" and product in self.product_info.get("replacements", {}):
                replacement = self.product_info["replacements"][product]
                print(f"  → Replaced by: {replacement}")
    
    def do_help(self, arg):
        """Display help information."""
        # Use the LLM name in the prefix
        prefix = f"[{self.llm.name}]"
        
        print(f"\n{prefix} Available commands:")
        print("  status    - Show current product information status")
        print("  exit/quit - Exit the chat interface")
        print("\nSample questions you can ask:")
        print("  What are the specifications of ProductA?")
        print("  Is ProductAccessory compatible with ProductB?")
        print("  How do I set up ProductB?")
        print("  What products do you offer?")


class ProductUpdateSimulator:
    """Simulates product updates by another process."""
    
    def __init__(self, db_path: str = "./demo_vector_db"):
        """Initialize the simulator."""
        self.db_path = db_path
        self.relationships_file = os.path.join(db_path, "product_relationships.json")
    
    def simulate_update(self):
        """Simulate a product update."""
        if not os.path.exists(self.relationships_file):
            print("No product relationships file found. Run the product replacement demo first.")
            return
        
        with open(self.relationships_file, 'r') as f:
            product_info = json.load(f)
        
        # Check if ProductA is already retired
        if "ProductA" in product_info.get("products", {}) and product_info["products"]["ProductA"].get("status") == "retired":
            print("ProductA is already retired. No update needed.")
            return
        
        print("Simulating product retirement...")
        
        # Update product status
        if "ProductA" in product_info.get("products", {}):
            product_info["products"]["ProductA"]["status"] = "retired"
            product_info["products"]["ProductA"]["retirement_date"] = datetime.now().isoformat()
        
        # Add replacement information
        if "replacements" not in product_info:
            product_info["replacements"] = {}
        product_info["replacements"]["ProductA"] = "ProductB"
        
        # Add to history
        if "history" not in product_info:
            product_info["history"] = []
        product_info["history"].append({
            "date": datetime.now().isoformat(),
            "action": "retirement",
            "product_id": "ProductA",
            "replacement_product_id": "ProductB"
        })
        
        # Save updated information
        with open(self.relationships_file, 'w') as f:
            json.dump(product_info, f, indent=2)
        
        print("Product update simulated: ProductA has been retired and replaced by ProductB")


def run_chat_interface():
    """Run the chat interface."""
    try:
        chat = ProductChatAssistant()
        chat.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your environment setup and try again.")


def run_update_simulator():
    """Run the update simulator after a delay."""
    simulator = ProductUpdateSimulator()
    
    # Wait for a random time between 15-30 seconds
    delay = random.randint(15, 30)
    print(f"Product update will be simulated in {delay} seconds...")
    time.sleep(delay)
    
    simulator.simulate_update()


if __name__ == "__main__":
    # Check if the database path exists
    if not os.path.exists("./demo_vector_db"):
        print("Database path not found. Please run the product replacement demo first.")
        exit(1)
    
    # Start the update simulator in a background thread
    update_thread = threading.Thread(target=run_update_simulator)
    update_thread.daemon = True
    update_thread.start()
    
    # Run the chat interface in the main thread
    run_chat_interface()