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
   git clone https://github.com/your-username/chatbot.git
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

## Project Structure
- `main.py`: Main application and chatbot logic
- `models/llm_engine.py`: LLM integration and text generation
- `applib.py`: Utility functions
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore rules

## License
[Specify your license here]
