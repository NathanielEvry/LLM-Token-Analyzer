import requests
import json
import sys
import os
import signal

# Written by Nathaniel Evry 2025-03-18

# Configuration (move to the top for easy tweaking)
# MODEL_NAME = "qwen2.5-0.5b-instruct"  # Change this to sweep other models
MODEL_NAME = "gemma-3-1b-it"  # Change this to sweep other models
START_ID = 1
END_ID = 260000
OUTPUT_FILE = f"token_mappings_{MODEL_NAME}.json"  # File to save/load token mappings

class TokenSweeper:
    def __init__(self, model_name):
        self.model = model_name
        # self.api_url = "http://localhost:1234/v1/completions"
        self.api_url = "http://10.0.0.195:1234/v1/completions"
        self.headers = {"Content-Type": "application/json"}
        self.save_interval = 1000  # Save every 100 tokens
        self.token_counter = 0  # Counter to track tokens since last save
        self.token_index = self._load_existing_mappings()
        self.should_exit = False  # Flag to handle graceful exit

        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, signum, frame):
        """Handle Ctrl+C by setting the exit flag and saving progress."""
        print("\nCtrl+C detected. Exiting gracefully...")
        self.should_exit = True

    def sweep_token_ids(self, start_id, end_id):
        """Sweep token IDs and save mappings to a file."""
        for token_id in range(start_id, end_id + 1):
            if self.should_exit:
                break  # Exit the loop if Ctrl+C was pressed

            if str(token_id) in self.token_index:
                print(f"Skipping ID {token_id} (already mapped)")
                continue

            payload = {
                "model": self.model,
                "prompt": " ",  # Use a single space as the prompt
                "temperature": 0.1,
                "max_tokens": 1,
                "logit_bias": {str(token_id): 100},
                "stream": True  # Enable streaming
            }

            try:
                with requests.post(self.api_url, headers=self.headers, json=payload, stream=True) as response:
                    if response.status_code == 200:
                        # Read the first chunk of the stream
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_str = chunk.decode("utf-8").strip()
                                if chunk_str.startswith("data: "):
                                    chunk_str = chunk_str[6:]

                                chunk_data = json.loads(chunk_str)
                                token = chunk_data["choices"][0]["text"].strip()
                                # Ensure only one token is returned
                                if token:
                                    token = token.split()[0]  # Take the first token if multiple are returned
                                    token_bytes = token.encode("utf-8")
                                    self.token_index[str(token_id)] = {
                                        "character": token,
                                        "bytes": list(token_bytes)  # Store raw bytes as a list of integers
                                    }
                                    print(f"[{token}]", end='')
                                    self.token_counter += 1
                                    if self.token_counter >= self.save_interval:
                                        self._save_mappings()  # Save progress every 100 tokens
                                        self.token_counter = 0  # Reset the counter
                                    break  # Stop after the first token
                    else:
                        print(f"Error for token ID {token_id}: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"!! ID {token_id}: {e}")

        # Final save in case there are remaining tokens not saved yet
        if not self.should_exit or self.token_counter > 0:
            self._save_mappings()

    def _load_existing_mappings(self):
        """Load existing token mappings from file."""
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_mappings(self):
        """Save token mappings to file, sorted by token ID."""
        # Sort the token_index by token ID (convert keys to integers for sorting)
        sorted_token_index = {k: self.token_index[k] for k in sorted(self.token_index, key=int)}
        with open(OUTPUT_FILE, "w") as f:
            json.dump(sorted_token_index, f, indent=2)
        print(f"\n{len(sorted_token_index)} IDs: {OUTPUT_FILE}")


# Main execution
if __name__ == "__main__":
    # Override defaults with command-line arguments
    if len(sys.argv) > 1:
        MODEL_NAME = sys.argv[1]
    if len(sys.argv) > 2:
        START_ID = int(sys.argv[2])
    if len(sys.argv) > 3:
        END_ID = int(sys.argv[3])

    # Create an instance of TokenSweeper
    sweeper = TokenSweeper(MODEL_NAME)

    # Perform the sweep
    print(f"Sweeping token IDs {START_ID} to {END_ID} for model: {MODEL_NAME}")
    sweeper.sweep_token_ids(START_ID, END_ID)

    print(f"Token mappings saved to {OUTPUT_FILE}")