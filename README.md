# Bissi Chatbot

Bissi is a conversational AI chatbot built with Python and Mistral 7B LLM.

## Features
- Natural language processing with Mistral 7B model
- French language support
- Interactive command-line interface
- Conversation history tracking

## Prerequisites
- Python 3.8+
- Required Python packages (see `requirements.txt`)
- Mistral 7B model file (GGUF format)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/goldensam777/ChatBot-b000.git
   cd chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Mistral 7B model (GGUF format) and place it in the `models/` directory.

## Usage

Run the chatbot:
```bash
python main.py
```

### Available Commands
- Type your message to chat with Bissi
- Type `help` to see available commands
- Type `clear` to start a new conversation
- Type `quit` or `exit` to exit the program

## Project Evolution

### Initial Version
I started with a simple language model implementation using **TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf**. This 1.1 billion parameter model, while not as powerful as larger models, provides a great starting point for motivation and testing. The chatbot currently supports English.

### Project Structure
```
chatbot
├── main.py                 # Main application and chatbot logic
├── models/
│   ├── llm_engine.py      # LLM integration and text generation
│   ├── tiny_config.json   # Configuration for TinyLlama
│   └── TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
├── applib.py              # Utility functions
└── requirements.txt       # Project dependencies
```

### Dependencies
The core dependency is `llama-cpp-python` for GGUF model support:

```
llama-cpp-python>=0.2.23  # Python interface for GGUF models (compatible with LLaMA, Mistral, etc.)
```

**Note:** The language model file (e.g., TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf) must be downloaded separately and placed in the `models/` directory.

### Next Steps
I've successfully tested the system with more advanced models and plan to continue improving the implementation.
- `.gitignore`: Git ignore rules

## License
[Specify your license here]
