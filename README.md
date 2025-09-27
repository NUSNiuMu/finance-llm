# Financial Chatbot System
A comprehensive financial chatbot system with multiple AI models and rule-based responses. Built on TensorFlow, PyTorch, and Hugging Face Transformers.

## 🚀 Features

- **Multiple AI Models**: Support for DialoGPT, ChatGLM, and custom fine-tuned models
- **Rule-based System**: Intelligent financial domain knowledge extraction
- **Hybrid Approach**: Combines dataset matching, rule-based responses, and AI generation
- **Financial Domain**: Specialized in investment, loans, insurance, banking services
- **Interactive Chat**: Real-time conversation with context awareness

## 📁 Project Structure

```
├── enhanced_financial_chatbot.py    # Main enhanced chatbot with AI models
├── chatbot.py                       # Original Reddit-based chatbot
├── finetune_dialoGPT_finance.py     # Model fine-tuning script
├── model.py                         # TensorFlow model definitions
├── utils.py                         # Utility functions
├── models/                          # Model checkpoints and weights
│   ├── generative-finetune/         # Fine-tuned DialoGPT model
│   └── reddit/                      # Reddit-trained character-level model
└── data/                           # Training datasets
```

## 🤖 Enhanced Financial Chatbot

The `enhanced_financial_chatbot.py` is the main chatbot system that combines multiple AI approaches:

### Key Features:
- **Multi-Model Support**: DialoGPT, ChatGLM, and custom fine-tuned models
- **Intelligent Rule Extraction**: Automatically builds rules from financial datasets
- **Hybrid Response Generation**: Combines dataset matching, rule-based responses, and AI generation
- **Financial Domain Expertise**: Specialized knowledge in investment, loans, insurance, banking

### Sample Conversation:

```
$ python enhanced_financial_chatbot.py
Loading financial dataset and building rule system...
Loading financial dataset...
Successfully loaded financial dataset with 518185 training samples
Extracting rules from dataset using 100000 samples...
Successfully extracted rules from 99947 Q&A pairs
============================================================
Welcome to Enhanced Financial Assistant!
I can provide professional consultation on investment, loans, insurance, banking services and more.
Type '--help' to view detailed usage instructions
Type 'quit' or 'exit' to exit the program
============================================================

You: What are good low-risk investment options?
Assistant: Based on your question, I found a similar question:

Question: What are some low-risk investment options for beginners?

Answer: For low-risk investments, consider:
1. Money market funds - provide stability with modest returns
2. Government bonds - backed by government credit
3. High-yield savings accounts - FDIC insured
4. Certificate of Deposits (CDs) - fixed interest rates
5. Treasury bills - short-term government securities

You: How do I apply for a mortgage?
Assistant: I found a related financial question:

Question: What is the process for applying for a mortgage?

Answer: The mortgage application process typically includes:
1. Pre-qualification - assess your financial situation
2. Documentation - income statements, tax returns, bank statements
3. Credit check - review your credit history and score
4. Property appraisal - assess the property value
5. Underwriting - final review and approval decision
6. Closing - sign documents and complete the transaction
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (for large models)

### Install Dependencies
```bash
pip install torch transformers datasets numpy pandas
pip install tensorflow scikit-learn matplotlib seaborn
pip install requests tqdm
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/NUSNiuMu/finance-llm.git
cd finance-llm

# Run the enhanced financial chatbot
python enhanced_financial_chatbot.py
```

## 🚀 Usage

### Enhanced Financial Chatbot (Recommended)
```bash
python enhanced_financial_chatbot.py
```

**Features:**
- Automatic financial dataset loading (500k+ samples)
- Intelligent rule extraction from data
- Multi-model support (DialoGPT, ChatGLM, etc.)
- Hybrid response generation
- Real-time conversation with context

**Commands:**
- `--help`: Show detailed usage instructions
- `--pretrained`: Toggle AI model usage
- `--reset`: Reset conversation history
- `quit`/`exit`: Exit program

### Original Reddit Chatbot
```bash
python chatbot.py
```

**Features:**
- Character-level text generation
- Reddit-trained conversational model
- Configurable sampling parameters

## 🔧 Model Training

### Fine-tune DialoGPT for Finance
```bash
python finetune_dialoGPT_finance.py
```

This script will:
- Load the Finance-Instruct-500k dataset
- Fine-tune DialoGPT-medium on financial conversations
- Save the model to `models/generative-finetune/`

### Download Pre-trained Models

**Reddit Model (Original):**
- Download [pre-trained Reddit model](https://drive.google.com/uc?id=1rRRY-y1KdVk4UB5qhu7BjQHtfadIOmMk&export=download) (2.3 GB)
- Extract to `models/reddit/` directory
- Warning: Contains unfiltered Reddit content

**Hugging Face Models:**
The enhanced chatbot automatically downloads models from Hugging Face:
- `microsoft/DialoGPT-medium` (default)
- `THUDM/chatglm2-6b` (recommended for Chinese)
- `Qwen/Qwen-7B-Chat` (Alibaba's model)

## ⚙️ Configuration

### Enhanced Chatbot Parameters
The enhanced chatbot supports various configuration options:

**Model Selection:**
```python
# Initialize with different models
chatbot = EnhancedFinancialChatbot(
    model_name="THUDM/chatglm2-6b",  # ChatGLM2-6B
    model_backend="hf"
)
```

**Dataset Size:**
```python
# Load different amounts of training data
chatbot.load_financial_dataset(sample_size=50000)  # 50k samples
```

### Original Chatbot Parameters
The original Reddit chatbot supports these parameters:

- **beam_width**: Beam search width (default: 2)
- **temperature**: Sampling temperature (default: 1.0)
- **top-n**: Top-n filtering (disabled by default)
- **relevance**: Relevance weighting (disabled by default)

**Runtime Configuration:**
```
$ python chatbot.py
Creating model...
Restoring weights...

> --temperature 1.3
[Temperature set to 1.3]

> --relevance 0.3
[Relevance set to 0.3]

> --beam_width 5
[Beam width set to 5]

> --reset
[Model state reset]
```

## 📊 Performance

### Enhanced Chatbot
- **Dataset**: 500k+ financial Q&A pairs
- **Response Time**: < 2 seconds (CPU), < 0.5 seconds (GPU)
- **Accuracy**: High for financial domain questions
- **Memory**: ~2GB RAM for DialoGPT-medium

### Original Chatbot
- **Training Data**: Reddit comments
- **Model Size**: ~2.3GB
- **Response Quality**: Variable (unfiltered content)
- **Speed**: Fast character-level generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Finance-Instruct-500k Dataset](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k)
- [DialoGPT](https://github.com/microsoft/DialoGPT)
- [ChatGLM](https://github.com/THUDM/ChatGLM-6B)

