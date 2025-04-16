"""
Simple RAG Chat Interface with Live Updates

This script provides a self-contained text-based chat interface that:
1. Reads from the same JSON knowledge base as the RAG system
2. Simulates queries about products with live updates
3. Monitors for changes to the product information
"""

import os
import json
import time
import threading
import cmd
import random
from datetime import datetime

class SimpleProductChat(cmd.Cmd):
    """
    Interactive chat interface for querying product information.
    This doesn't use external dependencies to ensure it works in all environments.
    """
    
    intro = """
==============================================
    Product Information Chat Assistant
==============================================
Ask questions about our products to get information.
Type 'help' for commands, 'exit' to quit.
"""
    prompt = "\nYou: "
    
    def __init__(self, db_path: str = "./demo_vector_db"):
        """Initialize the chat interface."""
        super().__init__()
        
        self.db_path = db_path
        self.relationships_file = os.path.join(db_path, "product_relationships.json")
        
        # Load product info
        self._load_product_info()
        
        # Information about each product (hardcoded for demonstration)
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
                    
                    print("\n\n[System] Product information has been updated!")
                    
                    # Print a summary of changes
                    retired_products = [p for p, info in self.product_info["products"].items() 
                                      if info.get("status") == "retired"]
                    
                    if retired_products:
                        print(f"[System] Retired products: {', '.join(retired_products)}")
                        for product in retired_products:
                            replacement = self.product_info["replacements"].get(product)
                            if replacement:
                                print(f"[System] {product} has been replaced by {replacement}")
                    
                    active_products = [p for p, info in self.product_info["products"].items() 
                                     if info.get("status") == "active"]
                    if active_products:
                        print(f"[System] Active products: {', '.join(active_products)}")
                    
                    print(f"\nYou: ", end="", flush=True)
                    
                    self.last_check_time = time.time()
    
    def _get_product_status(self, product_id):
        """Get the status of a product."""
        if product_id in self.product_info.get("products", {}):
            return self.product_info["products"][product_id].get("status")
        return None
    
    def _get_replacement(self, product_id):
        """Get replacement for a retired product."""
        return self.product_info.get("replacements", {}).get(product_id)
    
    def _generate_response(self, question):
        """Generate a response based on the question and product knowledge."""
        # Check for product mentions
        question_lower = question.lower()
        
        # First, handle questions about specific products
        for product_id in self.product_details:
            if product_id.lower() in question_lower:
                # Check product status
                status = self._get_product_status(product_id)
                
                # If product is retired, handle accordingly
                if status == "retired":
                    replacement = self._get_replacement(product_id)
                    
                    if replacement:
                        return (f"{product_id} has been retired and replaced by {replacement}. "
                                f"We recommend using {replacement} instead.\n\n"
                                f"If you're interested in {replacement}, it offers: "
                                f"{self.product_details.get(replacement, {}).get('specs', 'improved specifications')}")
                    else:
                        return f"{product_id} has been retired. We no longer offer this product."
                
                # Handle different types of questions about active products
                if "specifications" in question_lower or "specs" in question_lower:
                    return f"{product_id} specifications: {self.product_details[product_id].get('specs')}"
                
                elif "setup" in question_lower or "install" in question_lower:
                    return f"Setup instructions for {product_id}:\n{self.product_details[product_id].get('setup', 'No setup instructions available.')}"
                
                elif "troubleshoot" in question_lower or "problem" in question_lower:
                    return f"Troubleshooting for {product_id}:\n{self.product_details[product_id].get('troubleshooting', 'No troubleshooting information available.')}"
                
                # Default product information
                return f"Information about {product_id}:\n{self.product_details[product_id].get('specs')}"
        
        # Handle compatibility questions
        if "compatible" in question_lower or "work with" in question_lower:
            for product_id in self.product_details:
                if product_id.lower() in question_lower and "ProductAccessory" in question_lower:
                    # Get current compatibility info
                    compatibility = self.product_details["ProductAccessory"].get("compatibility", {}).get(product_id, "No compatibility information available.")
                    
                    # Check if product is retired
                    status = self._get_product_status(product_id)
                    if status == "retired":
                        replacement = self._get_replacement(product_id)
                        if replacement:
                            return (f"ProductAccessory was designed for {product_id}, which has been retired and replaced by {replacement}. "
                                   f"{self.product_details['ProductAccessory'].get('compatibility', {}).get(replacement, 'Limited compatibility information available for the new product.')}")
                    
                    return f"Compatibility information: {compatibility}"
            
            # Generic compatibility response
            return "Please specify which products you'd like compatibility information about."
        
        # Handle general product queries
        if "products" in question_lower or "offer" in question_lower:
            active_products = [p for p, info in self.product_info.get("products", {}).items() 
                             if info.get("status") == "active"]
            
            if active_products:
                return f"We currently offer the following products: {', '.join(active_products)}"
            else:
                return "I don't have information about our current product lineup."
        
        # Default response
        return "I'm not sure how to answer that question. Try asking about a specific product or its features."
    
    def default(self, line):
        """Process user input as a question to the system."""
        if line.lower() in ('exit', 'quit', 'bye'):
            return self.do_exit()
        
        try:
            # Generate a response
            answer = self._generate_response(line)
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
        print("\nCurrent Product Information:")
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
        print("\nAvailable commands:")
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
        chat = SimpleProductChat()
        chat.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")


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
    