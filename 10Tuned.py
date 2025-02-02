import os
import re
import ollama
import subprocess
import sys
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
import psutil
import json
import threading
import queue

# Constants and Configuration
MAX_CONSECUTIVE_FAILURES = 6
MAX_SUCCESSFUL_EXECUTIONS = 6
MAX_UNCHANGED_ITERATIONS = 6
ITERATIONS = 7
OUTPUT_FOLDER = 'output'
RAW_DATA_FOLDER = 'raw-data'
DATA_SAMPLE_LINES = 4

# LLM Models
CODE_MODELS = [
    'codellama:13b',
    'deepseek-coder-v2',
    'codegeex4',
]

REASONING_MODELS = [
    'llama3.1',
    'deepseek-llm'
]

current_reasoning_model_index = 0
current_code_model_index = 0

# Initialize Logging
logging.basicConfig(
    filename='host_code.log',
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Create Necessary Directories
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

# Initialize Variables
code = ''
previous_code = None
unchanged_iterations = 0
consecutive_failures = 0
categorical_columns = []
target_variable = 'Exited'  # Default target variable, adjust as needed

# Statistics Tracking
stats = {
    'total_iterations': 0,
    'successful_executions': 0,
    'warnings': 0,
    'errors': 0,
    'code_changes': 0,
    'execution_times': [],
    'memory_usage': [],
    'cpu_usage': [],
    'error_types': {},
    'performance_metrics': []
}

process = psutil.Process(os.getpid())

def beep(times=1):
    if sys.platform.startswith('darwin'):
        for _ in range(times):
            subprocess.run(['osascript', '-e', 'beep'])
    elif sys.platform.startswith('win'):
        import winsound
        for _ in range(times):
            winsound.MessageBeep()
    else:
        for _ in range(times):
            print('\a')

def create_and_activate_venv():
    """Create and activate a virtual environment."""
    venv_dir = os.path.join(OUTPUT_FOLDER, 'venv')
    if not os.path.exists(venv_dir):
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
    if os.name == 'nt':
        pip_executable = os.path.join(venv_dir, 'Scripts', 'pip.exe')
        python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe')
    else:
        pip_executable = os.path.join(venv_dir, 'bin', 'pip')
        python_executable = os.path.join(venv_dir, 'bin', 'python')
    return pip_executable, python_executable

def extract_code(llm_output):
    """Extract code blocks from LLM output."""
    code_pattern = r'```(?:python)?\n(.*?)```'
    code_matches = re.findall(code_pattern, llm_output, re.DOTALL)
    if code_matches:
        code = '\n'.join(code_matches)
    else:
        code = llm_output.strip()
    return code.strip()

def get_data_sample(file_path, num_lines=5):
    """Get a sample of the data for prompting."""
    try:
        df = pd.read_csv(file_path, nrows=num_lines)
        return df.to_csv(index=False)
    except Exception as e:
        logging.error(f"Error reading data sample: {e}")
        return ""

def save_iteration_data(llm_name, prompt, response, iteration, error_output):
    """Save data for each iteration for debugging."""
    llm_folder = os.path.join(RAW_DATA_FOLDER, llm_name)
    os.makedirs(llm_folder, exist_ok=True)
    combined_filename = os.path.join(llm_folder, f'iteration_{iteration}.txt')
    with open(combined_filename, 'w', encoding='utf-8') as combined_file:
        combined_file.write(
            f"Prompt:\n{prompt}\n\nError Output:\n{error_output}\n\nResponse:\n{response}"
        )

def generate_initial_code():
    """Generate initial code using the reasoning and code models."""
    global code
    data_sample = get_data_sample('xdata.csv', DATA_SAMPLE_LINES)
    if not data_sample:
        logging.error("No data sample available. Exiting.")
        sys.exit(1)

    df_full = pd.read_csv('xdata.csv')
    categorical_columns = get_categorical_columns(df_full)
    filename = 'xdata.csv'

    reasoning_prompt = (
        f"Using the following data sample from '{filename}':\n\n{data_sample}\n\n"
        f"Analyze the data and suggest the most suitable machine learning models and plot it. "
        f"Provide a one-line very short description about the choice of machine learning models. "
        f"If multiple models can be chosen, choose them. "
        f"Also, state which file contains the actual data, i.e., '{filename}'."
    )
    reasoning_system_prompt = "Recommend the suitable ML model based on the provided data."
    reasoning_model, code_model = get_current_models()

    reasoning_response = ollama.generate(
        model=reasoning_model,
        prompt=reasoning_prompt,
        system=reasoning_system_prompt
    )
    analysis = reasoning_response['response'].strip()
    save_iteration_data(reasoning_model, reasoning_prompt, analysis, 'initial', '')

    # Extract recommended model from analysis
    recommended_model = extract_recommended_model(analysis)
    if not recommended_model:
        logging.warning("No model was recommended by the LLM. Using default model 'LogisticRegression'.")
        recommended_model = 'LogisticRegression'

    # Prepare code generation prompt with logging instructions
    code_prompt = (
        f"Based on the recommended machine learning model ({recommended_model}), write a Python program to load '{filename}', "
        f"perform preprocessing (including encoding categorical variables: {', '.join(categorical_columns)} using pd.get_dummies with appropriate parameters to avoid deprecation warnings), "
        f"split the data into training and testing sets, and apply the model. "
        f"Include code for evaluating and plotting the results. "
        f"Add logging statements at the beginning and end of functions, before and after important operations, and during exception handling. "
        f"Use the Python 'logging' module configured to log in JSON format. Do not include any comments in the code. "
        f"Actual data in file '{filename}' is not limited to sample data; the actual data file is large. Use the following data sample (10 rows) for reference:\n\n"
        f"{data_sample}"
    )
    code_system_prompt = "Only write the code without any comments or explanations."
    code_response = ollama.generate(
        model=code_model,
        prompt=code_prompt,
        system=code_system_prompt
    )
    llm_output = code_response['response'].strip()
    save_iteration_data(code_model, code_prompt, llm_output, 'initial', '')

    beep(1)
    code = extract_code(llm_output)
    return code, recommended_model, analysis

def extract_recommended_model(analysis_text):
    """Extract the recommended model from the analysis text."""
    model_match = re.search(r'is the ([A-Za-z\s]+) model', analysis_text)
    if model_match:
        return model_match.group(1).strip().replace(" ", "")
    return None

def get_categorical_columns(df):
    """Identify categorical columns in the DataFrame."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def validate_schema(expected_columns, df):
    """Validate the schema of the DataFrame."""
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        return False, missing_columns
    return True, []

def validate_code(code, categorical_columns, target_variable):
    """Validate the generated code for essential elements."""
    required_imports = [
        'from sklearn.linear_model import LogisticRegression',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.metrics import accuracy_score',  # Adjusted for brevity
        'import logging'
    ]
    for imp in required_imports:
        if imp not in code:
            return False, f"Missing import: {imp}"

    # Check for proper encoding of categorical variables
    for col in categorical_columns:
        pattern = rf"pd\.get_dummies\(.*columns=\[['\"]{col}['\"]\].*\)"
        if not re.search(pattern, code):
            return False, f"Missing or incorrect encoding for categorical column: {col}"

    # Check for correct target variable
    target_pattern = rf"train_test_split\(.*,\s*df\['{target_variable}'\]"
    if not re.search(target_pattern, code):
        return False, f"Incorrect target variable used. Expected '{target_variable}'."

    # Static Code Analysis
    static_analysis_issues = run_static_analysis(code)
    if static_analysis_issues:
        return False, f"Static code analysis issues: {static_analysis_issues}"

    return True, ""

def run_static_analysis(code):
    """Run static analysis tools on the code."""
    # Save the code temporarily
    temp_code_path = 'temp_code.py'
    with open(temp_code_path, 'w') as f:
        f.write(code)

    # Run pylint
    try:
        result = subprocess.run(['pylint', temp_code_path, '--disable=all', '--enable=E,F'], capture_output=True, text=True)
        if result.returncode != 0:
            return result.stdout.strip()
    except Exception as e:
        logging.error(f"Static analysis failed: {e}")
        return f"Static analysis failed: {e}"
    finally:
        os.remove(temp_code_path)
    return ""

def fix_code(code, error_output, iteration, recommended_model="LogisticRegression"):
    """Fix the code using the LLM based on the error output."""
    beep(1)
    logging.info(f"Iteration {iteration}: Encountered error/warning: {error_output}")
    print(f"Iteration {iteration}: Encountered error/warning.")
    reasoning_model, code_model = get_current_models()

    data_sample = get_data_sample('xdata.csv', DATA_SAMPLE_LINES)
    original_code_prompt = (
        f"Based on the recommended machine learning model ({recommended_model}), write a Python program to load 'xdata.csv', "
        f"perform preprocessing (including encoding categorical variables: {', '.join(categorical_columns)} using pd.get_dummies with appropriate parameters to avoid deprecation warnings), "
        f"split the data into training and testing sets, and apply the model. "
        f"Add logging statements at the beginning and end of functions, before and after important operations, and during exception handling. "
        f"Use the Python 'logging' module configured to log in JSON format. Do not include any comments in the code. "
        f"Actual data in file 'xdata.csv' is not limited to sample data; the actual data file is large. Use the following data sample (10 rows) for reference:\n\n"
        f"{data_sample}"
    )

    llm_prompt = (
        f"The following Python code is producing an error or warning:\n\n"
        f"Error message:\n{error_output}\n\n"
        f"Code that needs fixing:\n"
        f"```python\n{code}\n```\n\n"
        f"The original task is:\n{original_code_prompt}\n"
        f"Ensure that the fixed code aligns with the original task and addresses the error. "
        f"Only provide the corrected code without any explanations or comments."
    )
    system_prompt = "Only write the corrected code without any comments or explanations."

    response = ollama.generate(
        model=code_model,
        prompt=llm_prompt,
        system=system_prompt
    )

    fixed_code = extract_code(response['response'].strip())
    save_iteration_data(code_model, llm_prompt, fixed_code, iteration, error_output)
    return fixed_code, llm_prompt

def get_current_models():
    """Get the current reasoning and code models."""
    reasoning_model = REASONING_MODELS[current_reasoning_model_index]
    code_model = CODE_MODELS[current_code_model_index]
    logging.info(f"Using Reasoning Model: {reasoning_model}, Code Model: {code_model}")
    return reasoning_model, code_model

def switch_to_next_model():
    """Switch to the next available model if current one fails."""
    global current_code_model_index, current_reasoning_model_index
    if current_code_model_index < len(CODE_MODELS) - 1:
        current_code_model_index += 1
        logging.info(f"Switched to next code model: {CODE_MODELS[current_code_model_index]}")
        return True
    elif current_reasoning_model_index < len(REASONING_MODELS) - 1:
        current_code_model_index = 0
        current_reasoning_model_index += 1
        logging.info(f"Switched to next reasoning model: {REASONING_MODELS[current_reasoning_model_index]}")
        return True
    else:
        logging.error("All available models have been exhausted.")
        return False

def identify_target_variable(df):
    """Identify the target variable in the DataFrame."""
    target_var = df.nunique().idxmin()
    return target_var

def main():
    """Main execution function."""
    global code, previous_code, unchanged_iterations, stats, consecutive_failures
    pip_executable, python_executable = create_and_activate_venv()

    # Upgrade pip and install initial dependencies
    subprocess.run([pip_executable, 'install', '--upgrade', 'pip'], check=True)
    initial_dependencies = ['pandas', 'matplotlib', 'scikit-learn', 'numpy', 'seaborn', 'pylint']
    subprocess.run([pip_executable, 'install'] + initial_dependencies, check=True)

    # Generate initial code
    code, recommended_model, analysis = generate_initial_code()
    if not code:
        logging.error("No code generated. Exiting.")
        return

    # Validate schema and code
    df_full = pd.read_csv('xdata.csv')
    categorical_columns = get_categorical_columns(df_full)
    target_variable = identify_target_variable(df_full)
    expected_columns = categorical_columns + [target_variable]
    is_valid_schema, missing_cols = validate_schema(expected_columns, df_full)
    if not is_valid_schema:
        error_message = f"Missing columns: {missing_cols}"
        fixed_code, _ = fix_code(code, error_message, 'initial', recommended_model=recommended_model)
        code = fixed_code
    is_valid, validation_msg = validate_code(code, categorical_columns, target_variable)
    if not is_valid:
        fixed_code, _ = fix_code(code, validation_msg, 'initial', recommended_model=recommended_model)
        code = fixed_code

    code_history = []

    for i in range(ITERATIONS):
        iteration_start_time = time.time()
        stats['total_iterations'] += 1
        beep(1)
        print('Iteration:', i)
        file_path = os.path.join(OUTPUT_FOLDER, f'generated_code_{i}.py')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        if previous_code == code:
            unchanged_iterations += 1
        else:
            unchanged_iterations = 0
            stats['code_changes'] += 1
        previous_code = code

        if unchanged_iterations >= MAX_UNCHANGED_ITERATIONS:
            print("No improvements detected. Stopping iterations.")
            logging.info("No improvements detected. Stopping iterations.")
            break

        error_output = ''

        # Resource Monitoring
        stats['memory_usage'].append(process.memory_info().rss / (1024 * 1024))
        stats['cpu_usage'].append(process.cpu_percent(interval=1))

        try:
            exec_start = time.time()
            result = subprocess.run([python_executable, file_path], check=True, capture_output=True, text=True)
            exec_end = time.time()
            execution_time = exec_end - exec_start
            stats['execution_times'].append(execution_time)

            if result.stderr:
                warnings = result.stderr.strip()
                logging.info(f"Iteration {i}: Warnings detected: {warnings}")
                stats['warnings'] += 1
                fixed_code, _ = fix_code(code, warnings, i, recommended_model=recommended_model)
                code = fixed_code
                consecutive_failures += 1

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    switched = switch_to_next_model()
                    if switched:
                        consecutive_failures = 0
                        code, recommended_model, analysis = generate_initial_code()
                    else:
                        print("All models have failed. Exiting.")
                        logging.error("All models have failed. Exiting.")
                        break
            else:
                print("Code executed successfully.", result.stdout)
                logging.info("Code executed successfully.")
                stats['successful_executions'] += 1
                beep(6)
                # Analyze successful execution and improve code
                code = analyze_successful_execution(result.stdout, code, recommended_model)
                logging.info(f"Iteration {i}: Analysis completed.")

                if stats['successful_executions'] > MAX_SUCCESSFUL_EXECUTIONS:
                    print("Sufficient successful executions achieved. Exiting.")
                    logging.info("Sufficient successful executions achieved. Exiting.")
                    break

        except subprocess.CalledProcessError as e:
            error_output = e.stderr.strip()
            error_type = extract_error_type(error_output)
            stats['errors'] += 1
            stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
            # Automated RCA
            root_cause = perform_root_cause_analysis(error_output)
            fixed_code, llm_prompt = fix_code(code, root_cause, i, recommended_model=recommended_model)
            code = fixed_code
            consecutive_failures += 1
            save_iteration_data(get_current_models()[1], llm_prompt, fixed_code, i, error_output)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                switched = switch_to_next_model()
                if switched:
                    consecutive_failures = 0
                    code, recommended_model, analysis = generate_initial_code()
                else:
                    print("All models have failed. Exiting.")
                    logging.error("All models have failed. Exiting.")
                    break

        except Exception as e:
            error_output = f"Unexpected error during execution: {e}\n\n"
            logging.error(error_output)
            stats['errors'] += 1
            consecutive_failures += 1
            # Automated RCA
            root_cause = perform_root_cause_analysis(error_output)
            fixed_code, llm_prompt = fix_code(code, root_cause, i, recommended_model=recommended_model)
            code = fixed_code
            save_iteration_data(get_current_models()[1], llm_prompt, fixed_code, i, error_output)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                switched = switch_to_next_model()
                if switched:
                    consecutive_failures = 0
                    code, recommended_model, analysis = generate_initial_code()
                else:
                    print("All models have failed. Exiting.")
                    logging.error("All models have failed. Exiting.")
                    break

        code_history.append({'iteration': i, 'code': code, 'error': error_output})
        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        stats['execution_times'].append(iteration_duration)

    else:
        print("Maximum iterations reached without successful execution.")
        logging.info("Maximum iterations reached without successful execution.")

    # Save statistics at the end
    save_stats(stats)
    generate_summary_report(stats)

def analyze_successful_execution(output, code, model):
    """Analyze successful execution and suggest improvements."""
    reasoning_model = REASONING_MODELS[current_reasoning_model_index]
    system_prompt = "Analyze the successful execution output and suggest improvements."
    prompt = (
        f"Code:\n```python\n{code}\n```\n"
        f"Execution Output:\n{output}\n"
        f"Please provide suggestions to improve the code or its execution based on the output."
    )
    response = ollama.generate(
        model=reasoning_model,
        prompt=prompt,
        system=system_prompt
    )
    analysis = extract_analysis(response['response'].strip())
    code = integrate_analysis_into_code(code, analysis)
    return code

def extract_analysis(response_text):
    """Extract analysis from the LLM response."""
    return response_text

def integrate_analysis_into_code(code, analysis):
    """Integrate LLM analysis into the code."""
    optimized_code = f"# Analysis: {analysis}\n" + code
    return optimized_code

def extract_error_type(error_message):
    """Extract error type from the error message."""
    match = re.search(r'(?P<error_type>\w+Error):', error_message)
    if match:
        return match.group('error_type')
    return 'UnknownError'

def perform_root_cause_analysis(error_message):
    """Perform automated root cause analysis on the error message."""
    # Implement a more sophisticated RCA here
    error_type = extract_error_type(error_message)
    suggestions = {
        'SyntaxError': 'Check for syntax errors in the code.',
        'NameError': 'A variable or function name is not defined.',
        'TypeError': 'An operation was applied to an object of inappropriate type.',
        # Add more error types and suggestions as needed
    }
    root_cause = suggestions.get(error_type, 'An error occurred.')
    return f"{error_type}: {root_cause}"

def analyze_error(error_message):
    """Analyze errors and generate helpful suggestions."""
    logging.debug(f"Analyzing error: {error_message}")
    # Implement error analysis logic here
    return error_message  # Return the processed error message

def save_stats(stats):
    """Save statistics to a JSON file."""
    with open('stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    logging.info("Statistics saved to stats.json")

def generate_summary_report(stats):
    """Generate a summary report from the statistics."""
    report = {
        'Total Iterations': stats['total_iterations'],
        'Successful Executions': stats['successful_executions'],
        'Errors': stats['errors'],
        'Warnings': stats['warnings'],
        'Code Changes': stats['code_changes'],
        'Average Execution Time': sum(stats['execution_times']) / len(stats['execution_times']) if stats['execution_times'] else 0,
        'Error Types': stats['error_types'],
    }
    with open('summary_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    logging.info("Summary report saved to summary_report.json")

if __name__ == "__main__":
    main()
    beep(6)
