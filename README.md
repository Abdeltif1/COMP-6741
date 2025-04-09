# COMP-6741


# AI Model Comparison and Chatbot System

A sophisticated chatbot system with multiple specialized agents, real-time model comparison capabilities, and a web interface for interaction and performance monitoring.

## Overview

This project consists of three main components:
1. Multi-agent chatbot system (main.py)
2. Web interface and API (app.py)
3. Model benchmarking tool (model_benchmark.py)

### Features

- Multiple specialized agents (General, Admission, AI)
- Real-time model comparison between Gemma and Llama
- Performance metrics and benchmarking
- Web interface for interaction
- Feedback collection and analysis
- Wikipedia integration for context
- Comprehensive evaluation metrics

## Installation

1. Clone the repository:

bash
git clone <repository-url>
cd <project-directory>


3. Install required dependencies:
pip install langchain-ollama langchain-core langchain-community flask pandas numpy scipy rouge-score nltk seaborn tqdm matplotlib

4. Install Ollama and download required models:


## Project Structure


├── main.py # Core chatbot functionality
├── app.py # Flask web application
├── model_benchmark.py # Model comparison tool
├── templates/ # HTML templates
│ ├── index.html
│ ├── metrics_dashboard.html
│ └── performance_evaluation.html
└── static/ # Static files (CSS, JS)


## Components

### main.py
- Implements the core chatbot functionality
- Manages multiple specialized agents
- Handles conversation memory and context
- Collects and stores feedback
- Tracks performance metrics

### app.py
- Provides web interface and REST API
- Visualizes performance metrics
- Handles user interactions
- Manages feedback collection
- Provides real-time statistics

### model_benchmark.py
- Compares Gemma and Llama models
- Generates performance metrics
- Creates visualization graphs
- Evaluates model accuracy and response time

## API Endpoints

- `/chat` - Send messages to the chatbot
- `/feedback` - Submit feedback for responses
- `/metrics` - Get performance metrics
- `/benchmarks` - Get model comparison data
- `/metrics-trends` - Get historical performance data
- `/ollama-stats` - Get Ollama usage statistics

## Evaluation Metrics

The system tracks multiple performance metrics:
- Response accuracy
- Coherence
- User satisfaction
- Response time
- Token usage
- Model comparison metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Notes

- Ensure Ollama is running before starting the application
- The system requires sufficient disk space for model storage
- Performance metrics are stored in CSV format
- Real-time metrics are available through the web interface
