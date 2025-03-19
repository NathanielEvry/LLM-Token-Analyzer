import requests
import json
import sys
import os
import signal
import time
from datetime import datetime

# Improved Token Sweeper by Nathaniel Evry and contributors
# This script systematically extracts the complete token vocabulary from an LLM

# Configuration 
MODEL_NAME = "gemma-3-1b-it"  # Default model name
START_ID = 1  # Default starting token ID
END_ID = 300000  # Default ending token ID
# API_URL = "http://localhost:1234/v1/completions"  # Local inference server
API_URL = "http://10.0.0.195:1234/v1/completions"  # Local inference server
SAVE_INTERVAL = 100  # Save progress every N tokens
OUTPUT_DIR = "token_mappings"  # Directory to store token mappings
RETRY_ATTEMPTS = 3  # Number of retry attempts for failed requests
RETRY_DELAY = 2  # Seconds to wait between retries

class TokenSweeper:
    def __init__(self, model_name):
        self.model = model_name
        self.api_url = API_URL
        self.headers = {"Content-Type": "application/json"}
        self.save_interval = SAVE_INTERVAL
        self.token_counter = 0
        self.should_exit = False
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Output file path
        self.output_file = os.path.join(OUTPUT_DIR, f"token_mappings_{model_name}.json")
        
        # Load existing mappings or initialize empty dict
        self.token_index = self._load_existing_mappings()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        
        # Statistics
        self.stats = {
            "start_time": datetime.now(),
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }

    def _handle_exit(self, signum, frame):
        """Handle exit signals by setting the flag to exit gracefully."""
        print("\n[!] Interrupt received. Finishing current token and exiting...")
        self.should_exit = True

    def sweep_token_ids(self, start_id, end_id):
        """Sweep token IDs and save mappings to a file."""
        print(f"[+] Starting token sweep for model: {self.model}")
        print(f"[+] Token range: {start_id} to {end_id}")
        print(f"[+] Output file: {self.output_file}")
        print(f"[+] Already mapped: {len(self.token_index)} tokens")
        
        try:
            for token_id in range(start_id, end_id + 1):
                if self.should_exit:
                    break
                
                # Skip already mapped tokens
                if str(token_id) in self.token_index:
                    self.stats["skipped"] += 1
                    continue
                
                # Process this token ID
                success = self._process_token_id(token_id)
                
                if success:
                    self.stats["successful"] += 1
                else:
                    self.stats["failed"] += 1
                
                self.stats["total_processed"] += 1
                
                # Save progress at intervals
                self.token_counter += 1
                if self.token_counter >= self.save_interval:
                    self._save_mappings()
                    self._print_stats()
                    self.token_counter = 0
        
        finally:
            # Final save and stats
            if self.token_counter > 0:
                self._save_mappings()
            
            self._print_stats(final=True)

    def _process_token_id(self, token_id):
        """Process a single token ID with retry logic."""
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # Display progress
                if attempt == 0:
                    print(f"\r[*] Processing ID {token_id}...", end="")
                
                payload = {
                    "model": self.model,
                    "prompt": " ",  # Single space prompt
                    "temperature": 0.1,
                    "max_tokens": 1,
                    "logit_bias": {str(token_id): 100},  # Force this token ID
                    "stream": True
                }
                
                with requests.post(self.api_url, headers=self.headers, json=payload, stream=True, timeout=10) as response:
                    if response.status_code == 200:
                        # Read the stream to get the token
                        for chunk in response.iter_lines():
                            if not chunk:
                                continue
                                
                            chunk_str = chunk.decode("utf-8").strip()
                            if chunk_str.startswith("data: "):
                                chunk_str = chunk_str[6:]
                            
                            # Skip the "[DONE]" message
                            if chunk_str == "[DONE]":
                                continue
                                
                            try:
                                chunk_data = json.loads(chunk_str)
                                token = chunk_data["choices"][0]["text"]
                                
                                # Store the token and its bytes
                                token_bytes = token.encode("utf-8")
                                self.token_index[str(token_id)] = {
                                    "character": token,
                                    "bytes": list(token_bytes)
                                }
                                
                                # Display the discovered token
                                print(f"\r[+] ID {token_id}: [{token}]" + " " * 20)
                                return True
                                
                            except json.JSONDecodeError:
                                continue
                    else:
                        # Non-200 response
                        if attempt == RETRY_ATTEMPTS - 1:
                            print(f"\r[!] Failed to process ID {token_id}: HTTP {response.status_code}")
                            return False
                
            except Exception as e:
                # Handle network errors, timeouts, etc.
                if attempt == RETRY_ATTEMPTS - 1:
                    print(f"\r[!] Error processing ID {token_id}: {str(e)}")
                    return False
            
            # Wait before retry
            time.sleep(RETRY_DELAY)
        
        return False  # Exhausted all retries

    def _load_existing_mappings(self):
        """Load existing token mappings from file."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Error loading existing mappings: {str(e)}")
                print("[!] Creating new mapping file.")
                return {}
        return {}

    def _save_mappings(self):
        """Save token mappings to file, sorted by token ID."""
        # Create a temp file for safety
        temp_file = f"{self.output_file}.tmp"
        
        try:
            # Sort the token_index by token ID
            sorted_token_index = {k: self.token_index[k] for k in sorted(self.token_index, key=int)}
            
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(sorted_token_index, f, indent=2, ensure_ascii=False)
            
            # Replace the original file with the temp file
            if os.path.exists(temp_file):
                if os.path.exists(self.output_file):
                    os.replace(temp_file, self.output_file)
                else:
                    os.rename(temp_file, self.output_file)
                    
        except Exception as e:
            print(f"\n[!] Error saving mappings: {str(e)}")
            if os.path.exists(temp_file):
                print(f"[!] Partial data was saved to {temp_file}")

    def _print_stats(self, final=False):
        """Print statistics about the token sweep."""
        elapsed = datetime.now() - self.stats["start_time"]
        elapsed_seconds = elapsed.total_seconds()
        
        total_tokens = len(self.token_index)
        tokens_per_second = self.stats["successful"] / elapsed_seconds if elapsed_seconds > 0 else 0
        
        if final:
            print("\n" + "=" * 50)
            print("TOKEN SWEEP COMPLETE")
            print("=" * 50)
        
        print(f"\n[*] Statistics:")
        print(f"    - Elapsed time: {elapsed}")
        print(f"    - Total tokens mapped: {total_tokens}")
        print(f"    - Tokens processed: {self.stats['total_processed']}")
        print(f"    - Successful: {self.stats['successful']}")
        print(f"    - Failed: {self.stats['failed']}")
        print(f"    - Skipped (already mapped): {self.stats['skipped']}")
        print(f"    - Processing rate: {tokens_per_second:.2f} tokens/second")
        
        if not final:
            print("\nContinuing token sweep...\n")

def parse_args():
    """Parse command line arguments."""
    args = {
        "model_name": MODEL_NAME,
        "start_id": START_ID,
        "end_id": END_ID
    }
    
    if len(sys.argv) > 1:
        args["model_name"] = sys.argv[1]
    if len(sys.argv) > 2:
        args["start_id"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        args["end_id"] = int(sys.argv[3])
    
    return args

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and run the token sweeper
    sweeper = TokenSweeper(args["model_name"])
    sweeper.sweep_token_ids(args["start_id"], args["end_id"])

if __name__ == "__main__":
    main()
