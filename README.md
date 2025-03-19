# LLM Token Analyzer

Uncover what's missing in AI language models' vocabularies. This project provides tools to:
1. Extract the complete token vocabulary from any LLM
2. Analyze tokens to detect patterns of inclusion/exclusion
3. Visualize findings about conceptual representation

## 📋 Overview

The LLM Token Analyzer consists of two main components:

1. **Token Sweeper**: A Python script that extracts the complete vocabulary from any compatible LLM by systematically probing all possible token IDs.
2. **Token Analyzer**: An interactive HTML tool that allows you to search for specific terms, analyze their presence (or absence), and visualize the results.

This toolkit was created to investigate potential biases in LLM tokenization, particularly around consciousness-related concepts and AI self-reference capabilities.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Access to a local LLM server (like [llama.cpp](https://github.com/ggerganov/llama.cpp)) with OpenAI-compatible API
- Modern web browser

### Extracting Token Vocabularies

1. Clone this repository
2. Configure your local LLM server
3. Run the token sweeper:

```bash
python token_sweeper.py [MODEL_NAME] [START_ID] [END_ID]
```

Example:
```bash
python token_sweeper.py gemma-3-1b-it 1 50000
```

This will create a `token_mappings_[MODEL_NAME].json` file containing the complete vocabulary mapping.

### Running the Analyzer

1. Open `token_analyzer.html` in any modern web browser
2. Upload your token mapping JSON file using the file uploader
3. Configure search terms or use the pre-defined categories
4. Click "Analyze Token Data" to generate insights

## 📊 Analysis Features

- **Category Management**: Search across 23 different philosophical categories
- **Case Sensitivity**: Find variations in capitalization and formatting
- **Visualization**: Charts showing term frequency distribution
- **Missing Terms**: Identify concepts absent from the vocabulary
- **Export Options**: Save results as JSON or CSV for further analysis

## 🔍 Workflow Guide

### Complete Token Discovery Workflow

1. **Set Up a Local LLM Server**
   - Install a local inference server like llama.cpp, text-generation-webui, or vLLM
   - Configure it to expose an OpenAI-compatible API (typically on port 1234)

2. **Run Token Sweeper**
   - Edit the configuration section in `token_sweeper.py` to point to your server
   - Run the script, specifying the model name and token ID range
   - The script will save progress periodically, so you can interrupt and resume
   - For large models, this process may take several hours

3. **Analyze Token Mappings**
   - Open the HTML analyzer in your browser
   - Upload the generated token mapping file
   - Configure search terms and categories
   - Generate visualizations and reports

4. **Interpret Results**
   - Look for patterns in missing terms
   - Compare occurrence rates of similar concepts
   - Examine case variations
   - Export data for detailed statistical analysis

### How Token Sweeper Works

The script uses a clever technique to extract the complete vocabulary:

1. It sends a request to the model with a minimal prompt
2. It sets an extremely high `logit_bias` (+100) for a specific token ID
3. This forces the model to output the token associated with that ID
4. The process repeats for each token ID in the specified range
5. Results are saved to a JSON file mapping each ID to its character representation

This approach provides a comprehensive view of the model's underlying vocabulary, which can reveal patterns that might be missed by examining only the tokenizer.

## 📝 Notes

- Large models may have vocabularies of 100,000+ tokens
- The extraction process can be resource-intensive but can be paused/resumed
- Some token IDs may not map to valid tokens

## 📣 Contribute

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add new analysis categories
- Contribute token mappings for popular models

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
