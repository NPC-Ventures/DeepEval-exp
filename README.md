# DeepEval

An evaluation framework for meeting summarization using DeepEval with support for both cloud and local (Ollama) backends.

## Project Structure

```
DeepEval/
├── evaluator.py                 # Main evaluator
├── test_app.py                  # Main test application
├── meeting_summarizer/          # Meeting summarizer module
│   ├── evaluator.py             # Meeting summarizer evaluator
│   ├── meeting_summarizer.py    # Cloud-based meeting summarizer
│   ├── meeting_summarizer_local.py # Local Ollama-based summarizer
│   ├── test_app.py              # Cloud test application
│   ├── test_app_local.py        # Local test application
│   └── transcripts/             # Sample meeting transcripts
└── requirements.txt             # Project dependencies
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory with your API credentials:

```bash
cp .env.example .env  # If you have an example file
```

Or create `.env` manually with the required environment variables:
```
OPENAI_API_KEY=your_openai_api_key
```

## Running the Application

### Cloud-Based Evaluation

#### Run DeepEval Tests
```bash
deepeval test run test_app.py
```

### Local Evaluation with Ollama

#### 1. Start Ollama
```bash
ollama serve
```

In a new terminal, pull a model (e.g., Llama 2):
```bash
ollama pull llama2
```

#### 2. Set Environment for Local Mode
```bash
deepeval set-ollama
```

#### 3. Run DeepEval Tests with Local Backend
```bash
deepeval test run test_app.py
```

### Switching Between Cloud and Local

#### Switch from Local to Cloud
```bash
deepeval unset-ollama
```

Then run your cloud-based test:
```bash
deepeval test run test_app.py
```

## Usage Examples

### Meeting Summarizer

The meeting summarizer module provides both cloud and local implementations:

- **Cloud Version**: Uses OpenAI LLMs via LangChain
- **Local Version**: Uses Ollama for local LLM inference

## Dependencies

Key packages used in this project:

- `deepeval` - Evaluation framework
- `langchain-ollama` - Ollama integration for LangChain
- `langchain-openai` - OpenAI integration for LangChain
- `llama-index` - Data indexing and querying
- `python-jose` - JWT token handling
- `pymongo` - MongoDB integration
- `docling` - Document processing

See `requirements.txt` for the complete list of dependencies.

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running with `ollama serve`
- Check that the Ollama service is accessible at the default port (usually 11434)
- Verify the model is downloaded with `ollama list`

### API Key Issues
- Ensure `.env` file exists in the root directory
- Verify all required API keys are set correctly
- Check that sensitive data is not committed to version control

### DeepEval Test Failures
- Run `deepeval test run test_app.py -v` for verbose output
- Check logs for detailed error messages
- Ensure the correct backend is configured (cloud or local)

## Notes

- Meeting transcripts are stored in `meeting_summarizer/transcripts/`


