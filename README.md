# AI-Powered Coding Agent

A sophisticated Python-based coding agent that leverages multiple Large Language Models (LLMs) to automatically generate, debug, and optimize machine learning code. This agent is designed to handle the entire ML pipeline from data preprocessing to model evaluation with minimal human intervention.

## Features

- **Multi-Model LLM Integration**
  - Uses multiple code generation models:
    - CodeLlama 13B
    - DeepSeek Coder v2
    - CodeGeex4
  - Employs reasoning models:
    - Llama 3.1
    - DeepSeek LLM

- **Intelligent Code Generation**
  - Automatically analyzes datasets to recommend suitable ML models
  - Generates complete ML pipeline code including:
    - Data preprocessing
    - Feature engineering
    - Model training
    - Evaluation
    - Results visualization

- **Advanced Error Handling**
  - Automatic error detection and resolution
  - Smart dependency management
  - Handles multiple types of errors:
    - Syntax errors
    - Runtime errors
    - Type errors
    - Missing dependencies
    - Warning resolutions

- **Robust Iteration System**
  - Maximum of 7 iterations per run
  - Handles up to 6 consecutive failures
  - Supports up to 6 successful executions
  - Monitors unchanged iterations (max 6)

- **Comprehensive Logging**
  - Detailed JSON-formatted logging
  - Performance metrics tracking
  - Resource usage monitoring (CPU, memory)
  - Error type statistics

## System Requirements

- Python 3.x
- Ollama (for LLM integration)
- Required Python packages:
  - pandas
  - matplotlib
  - scikit-learn
  - numpy
  - psutil

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and running with the required models:
```bash
ollama pull codellama:13b
ollama pull deepseek-r1:1.5b
ollama pull codegeex4
ollama pull llama3.1
ollama pull deepseek-llm
```

## Usage

1. Place your dataset in the project directory (default expected filename: 'xdata.csv')

2. Run the coding agent:
```bash
python 10Tuned.py
```

3. The agent will:
   - Analyze your dataset
   - Recommend suitable ML models
   - Generate and execute the code
   - Handle errors automatically
   - Save results in the 'output' directory

## Example Output

The agent is capable of generating production-ready machine learning code. Below is an example of a successfully generated ML pipeline that demonstrates the agent's capabilities in creating well-structured, robust code with proper error handling and logging.

Key features of the generated code:
- Proper import management
- Structured code organization with separate functions
- Comprehensive error handling with try-except blocks
- Detailed logging implementation
- Data preprocessing with categorical variable encoding
- Model training and evaluation with multiple metrics
- Clean and maintainable code structure

The agent automatically:
1. Identified the required libraries and imports
2. Detected categorical columns for preprocessing
3. Implemented proper error handling and logging
4. Set up a complete ML pipeline from data loading to evaluation
5. Added appropriate evaluation metrics (accuracy, ROC AUC, confusion matrix)

Generated Code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def load_data():
    try:
        logging.info("Loading data from xdata.csv")
        df = pd.read_csv('xdata.csv')
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    try:
        logging.info("Starting preprocessing")
        df = pd.get_dummies(df, columns=['Surname', 'Geography', 'Gender'], drop_first=True)
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(df):
    try:
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        logging.info(f"Model Evaluation - Accuracy: {accuracy}, ROC AUC Score: {roc_auc}")
        logging.info(f"Confusion Matrix: \n{conf_matrix}")
    except Exception as e:
        logging.error(f"Error evaluating the model: {e}")
        raise

def main():
    try:
        logging.info("Starting the main function")
        df = load_data()
        preprocessed_df = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(preprocessed_df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        logging.info("Main function completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

## Project Structure

```
.
├── output/               # Generated code and results
├── raw-data/            # Raw data from LLM interactions
├── 10Tuned.py          # Main coding agent script
├── host_code.log       # Detailed logging file
├── stats.json          # Statistics and metrics
└── summary_report.json # Execution summary
```

## Configuration

Key configuration parameters in `10Tuned.py`:

```python
MAX_CONSECUTIVE_FAILURES = 6
MAX_SUCCESSFUL_EXECUTIONS = 6
MAX_UNCHANGED_ITERATIONS = 6
ITERATIONS = 7
DATA_SAMPLE_LINES = 4
```

## Error Handling

The agent handles various types of errors:
- Missing module errors (automatically installs required packages)
- Syntax errors
- Runtime errors
- Type errors
- Index errors
- Future warnings
- SettingWithCopy warnings

## Monitoring and Analytics

The agent provides comprehensive monitoring through:
- Real-time execution logs
- Performance metrics
- Resource usage statistics
- Error type analysis
- Success rate tracking

## Development History

The project has evolved through multiple iterations, each adding new capabilities and improvements:

1. **Initial Versions (1-3)**
   - Basic code generation and error handling
   - Single model implementation
   - Simple logging system

2. **Middle Iterations (4-6)**
   - Introduction of multiple LLM models
   - Enhanced error handling
   - Implementation of model switching on failures
   - Improved logging with JSON format

3. **Advanced Versions (7-9)**
   - Multiple reasoning and code models integration
   - Sophisticated error pattern recognition
   - Performance metrics tracking
   - Resource usage monitoring

4. **Current Version (10Tuned)**
   - Multiple LLM model fallback system
   - Advanced error handling with pattern matching
   - Comprehensive logging and analytics
   - Robust iteration control
   - Smart dependency management
   - Auto-installation of missing packages
   - Performance optimization
   - Resource monitoring and statistics
   - Automated root cause analysis

Each iteration has contributed to making the agent more robust, efficient, and capable of handling complex ML tasks with minimal human intervention.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2024 Shariq Ayaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

 Shariq Ayaz 


## Acknowledgments

- Ollama team for LLM integration
- Various open-source ML libraries
- LLM model creators and maintainers 
