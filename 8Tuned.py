import os
import re
import ollama
import subprocess
import sys
import pandas as pd
import logging
import time
import json
import csv
import matplotlib.pyplot as plt

REASONING_MODEL = 'llama3.1'
CODE_MODEL = 'codellama:13b'
SECOND_CODE_MODEL = 'deepseek-coder-v2'
THIRD_CODE_MODEL = 'codegeex4'
FORTH_CODE_MODEL = 'starcoder2:15b'

def beep(times=1):
    """
    Triggers a beep sound across different operating systems.

    Parameters:
    times (int): Number of beep sounds to play.
    """
    if sys.platform.startswith('darwin'):
        # macOS
        for _ in range(times):
            subprocess.run(['osascript', '-e', 'beep'])
    elif sys.platform.startswith('win'):
        # Windows
        import winsound
        for _ in range(times):
            winsound.MessageBeep()
    else:
        for _ in range(times):
            print('\a') 

beep(1)

logging.basicConfig(filename='host_code.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

output_folder = 'output'
raw_data_folder = 'raw-data'
venv_dir = os.path.join(output_folder, 'venv')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(raw_data_folder, exist_ok=True)

iterations = 30
code = ''
previous_code = None  
unchanged_iterations = 0  
max_unchanged_iterations = 20  

data_sample_lines = 8  

categorical_columns = []

target_variable = 'Exited'

# --- Start of Added Statistics Tracking ---
# Initialize statistics
stats = {
    'total_iterations': 0,
    'successful_executions': 0,
    'warnings': 0,
    'errors': 0,
    'code_changes': 0,
    'execution_times': [],
    'memory_usage': [],
    'cpu_usage': [],
}

# Optional: Initialize psutil for resource usage
try:
    import psutil
    process = psutil.Process(os.getpid())
    stats['memory_usage'] = []
    stats['cpu_usage'] = []
except ImportError:
    psutil = None
# --- End of Added Statistics Tracking ---

def create_and_activate_venv():
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
    code_pattern = r'```(?:python)?\n(.*?)```'
    code_matches = re.findall(code_pattern, llm_output, re.DOTALL)
    if code_matches:
        code = '\n'.join(code_matches)
    else:
        code = llm_output.strip()
    
    return code.strip()

def get_data_sample(file_path, num_lines=5):
    try:
        df = pd.read_csv(file_path, nrows=num_lines)
        return df.to_csv(index=False)
    except Exception as e:
        logging.error(f"Error reading data sample: {e}")
        return ""

def save_raw_data(llm_name, prompt, response, iteration):
    """
    Saves each prompt and its corresponding response in a single file under the model's folder.

    Parameters:
    llm_name (str): The name of the LLM used.
    prompt (str): The prompt sent to the LLM.
    response (str): The response from the LLM.
    iteration (int/str): The current iteration or unique identifier for the files.
    """
    llm_folder = os.path.join(raw_data_folder, llm_name)
    os.makedirs(llm_folder, exist_ok=True)
    
    # Save the prompt and response together in a single file
    combined_filename = os.path.join(llm_folder, f'iteration_{iteration}.txt')
    with open(combined_filename, 'w', encoding='utf-8') as combined_file:
        combined_file.write(f"Prompt:\n{prompt}\n\nResponse:\n{response}")

def generate_initial_code():
    global code
    data_sample = get_data_sample('xdata.csv', data_sample_lines)
    if not data_sample:
        logging.error("No data sample available. Exiting.")
        sys.exit(1)

    df_full = pd.read_csv('xdata.csv')
    categorical_columns = get_categorical_columns(df_full)
    
    filename = 'xdata.csv'
    
    reasoning_prompt = (
        f"Using the following data sample from '{filename}':\n\n{data_sample}\n\n"
        f"Analyze the data and suggest the most suitable machine learning model and plot it"
        f"for the given dataset. Provide a one-line very short description about the choice of machine learning model. "
        f"Also, state which file contains the actual data, i.e., '{filename}'."
    )
    reasoning_system_prompt = "Recommend the suitable ML model based on the provided data."
    reasoning_model = REASONING_MODEL

    reasoning_response = ollama.generate(
        model=reasoning_model,
        prompt=reasoning_prompt,
        system=reasoning_system_prompt
    )
    analysis = reasoning_response['response'].strip()
  
    save_raw_data(reasoning_model, reasoning_prompt, analysis, 'initial')
    
    # --- Start of Modified Model Extraction ---
    # Updated regex to capture only the model name
    model_match = re.search(r'is the ([A-Za-z\s]+) model', analysis)
    if model_match:
        recommended_model = model_match.group(1).strip().replace(" ", "")
    else:
        recommended_model = None
    # --- End of Modified Model Extraction ---
    
    if not recommended_model:
        logging.warning("No model was recommended by the LLM. Using default model selection logic.")
        
        # recommended_model = "LogisticRegression"
    
    code_prompt = (
        f"Based on the recommended machine learning model ({recommended_model}), write a Python program to load '{filename}', "
        f"perform preprocessing (including encoding categorical variables: {', '.join(categorical_columns)} using pd.get_dummies with appropriate parameters to avoid deprecation warnings), "
        f"split the data into training and testing sets, and apply the model. "
        f"Include code for evaluating and plotting the results. Do not include any comments in the code. "
        f"Actual data in file '{filename}' is not limited to sample data; the actual data file is large. Use the following data sample (10 rows) for reference:\n\n"
        f"{data_sample}*"
    )

    code_system_prompt = "Only write the code without any comments or explanations."
    code_model = CODE_MODEL

    code_response = ollama.generate(
        model=code_model,
        prompt=code_prompt,
        system=code_system_prompt
    )
    llm_output = code_response['response'].strip()

    save_raw_data(code_model, code_prompt, llm_output, 'initial')

    beep(1)

    code = extract_code(llm_output)
    return code, recommended_model, analysis

def get_categorical_columns(df):
    """
    Dynamically identifies categorical columns in the DataFrame based on data types.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    list: A list of categorical column names.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_cols

def validate_schema(expected_columns, df):
    """
    Validates that the DataFrame contains all expected columns.
    
    Parameters:
    expected_columns (list): List of expected column names.
    df (pd.DataFrame): The DataFrame to validate.
    
    Returns:
    tuple: (is_valid (bool), missing_columns (list))
    """
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        return False, missing_columns
    return True, []

def validate_code(code, categorical_columns, target_variable):
    """
    Validates the generated code for necessary imports, encoding steps, and target variable usage.
    
    Parameters:
    code (str): The generated Python code.
    categorical_columns (list): List of categorical column names.
    target_variable (str): The name of the target variable.
    
    Returns:
    tuple: (is_valid (bool), validation_msg (str))
    """
    required_imports = [
        'from sklearn.linear_model import LogisticRegression',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve'
    ]
    for imp in required_imports:
        if imp not in code:
            return False, f"Missing import: {imp}"
    
    for col in categorical_columns:
        # Dynamic pattern matching based on actual column names
        pattern = rf"pd\.get_dummies\(df, columns=\[['\"]{col}['\"]\], drop_first=True\)"
        if not re.search(pattern, code):
            return False, f"Missing or incorrect encoding for categorical column: {col}"
    
    # Dynamic target variable pattern matching
    target_pattern = rf"train_test_split\(.*?,\s*df\['{target_variable}'\]"
    if not re.search(target_pattern, code):
        return False, f"Incorrect target variable used. Expected '{target_variable}'."
    return True, ""

def fix_code(code, error_output, iteration, recommended_model="LogisticRegression"):
    """
    Fixes the provided Python code based on the encountered error or warning by interacting with the LLM.

    Parameters:
    code (str): The current version of the Python code.
    error_output (str): The error or warning message captured from the code execution.
    iteration (int): The current iteration count.
    recommended_model (str): The machine learning model recommended for use.

    Returns:
    str: The corrected Python code.
    """
    beep(1)
    logging.info(f"Iteration {iteration}: Encountered error/warning: {error_output}")
    print(f"Iteration {iteration}: Encountered error/warning.")

    # Patterns to identify different error types
    module_not_found_pattern = r"No module named ['\"]([^'\"]+)['\"]"
    key_error_pattern = r"KeyError: ['\"]([^'\"]+)['\"]"
    syntax_error_pattern = r"SyntaxError: (.+)"
    attribute_error_pattern = r"AttributeError: (.+)"
    value_error_pattern = r"ValueError: (.+)"
    type_error_pattern = r"TypeError: (.+)"
    index_error_pattern = r"IndexError: (.+)"
    future_warning_pattern = r"FutureWarning: (.*)"
    setting_with_copy_pattern = r"SettingWithCopyWarning: (.*)"

    # Retrieve the DataFrame's column names
    df_columns = get_categorical_columns(pd.read_csv('xdata.csv'))
    df_columns_str = ', '.join(df_columns)

    # Extract the file and line number from the error output (if available)
    tb_regex = r'File "(.+?)", line (\d+),'
    tb_match = re.search(tb_regex, error_output)

    if tb_match:
        file_path = tb_match.group(1)
        line_number = int(tb_match.group(2))

        # Read the content of the .py file and extract the specific line
        try:
            with open(file_path, 'r') as file:
                file_content = file.readlines()
                error_line = file_content[line_number - 1].strip()  # Get the line causing the error
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            error_line = "Could not read the file content."
    else:
        error_line = "Line causing error could not be identified."

    # Check for ModuleNotFoundError
    module_match = re.search(module_not_found_pattern, error_output)
    if module_match:
        missing_module = module_match.group(1)
        pip_executable, _ = create_and_activate_venv()
        # Map common module names to their installable package names if necessary
        module_install_map = {
            'sklearn': 'scikit-learn',
            'numpy': 'numpy',
            'pandas': 'pandas',

        }
        install_package = module_install_map.get(missing_module, missing_module)
        subprocess.run([pip_executable, 'install', install_package], check=True)
        logging.info(f"Installed missing module: {install_package}")
        return code

    data_sample = get_data_sample('xdata.csv', data_sample_lines)

    # Check for KeyError
    key_error_match = re.search(key_error_pattern, error_output)
    if key_error_match:
        missing_key = key_error_match.group(1)
        llm_prompt = (
            f"The following Python code is producing a KeyError: '{missing_key}'.\n\n"
            f"```python\n{code}\n```\n\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"The dataset has the following columns: {df_columns_str}.\n"
            f"It seems that the column '{missing_key}' does not exist in the DataFrame. "
            f"Please verify the column names and ensure consistency between the dataset and the code. "
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Only provide the corrected code without any explanations or comments."
        )
    # Check for SyntaxError
    elif re.search(syntax_error_pattern, error_output):
        syntax_detail = re.search(syntax_error_pattern, error_output).group(1)
        llm_prompt = (
            f"The following Python code is producing a SyntaxError:\n\n"
            f"\n{syntax_detail}\n"
            f"```python\n{code}\n```\n\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Please fix the syntax error in the code. Only provide the corrected code without any explanations or comments."
        )
    # Check for AttributeError
    elif re.search(attribute_error_pattern, error_output):
        attribute_detail = re.search(attribute_error_pattern, error_output).group(1)
        llm_prompt = (
            f"The following Python code is producing an AttributeError:\n\n"
            f"\n{attribute_detail}\n"
            f"```python\n{code}\n```\n\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Please fix the attribute error in the code. Only provide the corrected code without any explanations or comments."
        )
    # Check for ValueError
    elif re.search(value_error_pattern, error_output):
        value_error_detail = re.search(value_error_pattern, error_output).group(1)
        llm_prompt = (
            f"The following Python code is producing a ValueError:\n\n"
            f"\n{value_error_detail}\n"
            f"```python\n{code}\n```\n\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Please fix the value error in the code. Only provide the corrected code without any explanations or comments."
        )
    # Check for TypeError
    elif re.search(type_error_pattern, error_output):
        type_error_detail = re.search(type_error_pattern, error_output).group(1)
        llm_prompt = (
            f"The following Python code is producing a TypeError:\n\n"
            f"\n{type_error_detail}\n"
            f"```python\n{code}\n```\n\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Please fix the type error in the code. Only provide the corrected code without any explanations or comments."
        )
    # Check for IndexError
    elif re.search(index_error_pattern, error_output):
        index_error_detail = re.search(index_error_pattern, error_output).group(1)
        llm_prompt = (
            f"The following Python code is producing an IndexError:\n\n"
            f"\n{index_error_detail}\n"
            f"```python\n{code}\n```\n\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Please fix the index error in the code. Only provide the corrected code without any explanations or comments."
        )
    # Check for FutureWarning
    elif re.search(future_warning_pattern, error_output):
        warning_detail = re.search(future_warning_pattern, error_output).group(1)
        llm_prompt = (
            f"The following Python code is producing a FutureWarning:\n\n"
            f"\n{warning_detail}\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"```python\n{code}\n```\n\n"
            f"Warning message:\n{error_output}\n\n"
            f"Please update the code to handle this warning appropriately, ensuring compatibility with future library versions. "
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Only provide the corrected code without any explanations or comments."
        )
    # Check for SettingWithCopyWarning
    elif re.search(setting_with_copy_pattern, error_output):
        warning_detail = re.search(setting_with_copy_pattern, error_output).group(1)
        llm_prompt = (
            f"The following Python code is producing a SettingWithCopyWarning:\n\n"
            f"\n{warning_detail}\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"```python\n{code}\n```\n\n"
            f"Warning message:\n{error_output}\n\n"
            f"Please modify the code to avoid this warning, ensuring proper DataFrame manipulation. "
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Only provide the corrected code without any explanations or comments."
        )
    else:
        # Generic error or warning handling
        llm_prompt = (
            f"The following Python code is producing an error or warning:\n\n"
            f"Error message:\n{error_output}\n\n at: {error_line}"
            f"```python\n{code}\n```\n\n"
            f"Message:\n{error_output}\n\n"
            f"The dataset includes categorical variables that may need encoding.\n"
            
            f"\nSome data from Dataset\n*: {data_sample}*"
            f"\nEnsure that the model is fitted to training data before prediction\n"
            f"Please fix the code to handle these issues. Only provide the corrected code without any explanations or comments."
        )

    # Define system prompt and model
    system_prompt = "Only write the corrected code without any comments or explanations."
    llm_model = CODE_MODEL

    response = ollama.generate(
        model=llm_model,
        prompt=llm_prompt,
        system=system_prompt
    )
    
    fixed_code = extract_code(response['response'].strip())

    save_raw_data(llm_model, llm_prompt, fixed_code, iteration)

    return fixed_code

def identify_target_variable(df):
    """
    Identifies the target variable in the DataFrame.
    This can be based on business logic or configuration.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    str: The name of the target variable.
    """
    target_var = df.nunique().idxmin()
    return target_var

def main():
    global code, previous_code, unchanged_iterations, stats
    pip_executable, python_executable = create_and_activate_venv()

    subprocess.run([pip_executable, 'install', '--upgrade', 'pip'], check=True)
    
    initial_dependencies = [
        'pandas',
        'matplotlib',
        'scikit-learn',
        'numpy',
        'seaborn'
    ]
    subprocess.run([pip_executable, 'install'] + initial_dependencies, check=True)

    code, recommended_model, analysis = generate_initial_code()

    if not code:
        logging.error("No code generated. Exiting.")
        return

    df_full = pd.read_csv('xdata.csv')
    categorical_columns = get_categorical_columns(df_full)
    target_variable = identify_target_variable(df_full)
    
    # Define expected columns based on categorical columns and target variable
    expected_columns = categorical_columns + [target_variable]
    
    is_valid_schema, missing_cols = validate_schema(expected_columns, df_full)
    if not is_valid_schema:
        error_message = f"Missing columns: {missing_cols}"
        code = fix_code(code, error_message, 'initial', recommended_model=recommended_model)
    
    is_valid, validation_msg = validate_code(code, categorical_columns, target_variable)
    if not is_valid:
        code = fix_code(code, validation_msg, 'initial', recommended_model=recommended_model)


    for i in range(iterations):
        
        iteration_start_time = time.time()
        stats['total_iterations'] += 1

        beep(1)
        print('Iteration:', i)
        file_path = os.path.join(output_folder, f'generated_code_{i}.py')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        if previous_code == code:
            unchanged_iterations += 1
        else:
            unchanged_iterations = 0
            stats['code_changes'] += 1
        previous_code = code

        if unchanged_iterations >= max_unchanged_iterations:
            print("No improvements detected. Stopping iterations.")
            logging.info("No improvements detected. Stopping iterations.")
            break

        # Track resource usage
        if psutil:
            stats['memory_usage'].append(process.memory_info().rss / (1024 * 1024))  # in MB
            stats['cpu_usage'].append(process.cpu_percent(interval=1))  # CPU usage percentage
        try:
            # Execute the generated code within the virtual environment
            exec_start = time.time()
            result = subprocess.run([python_executable, file_path], check=True, capture_output=True, text=True)
            exec_end = time.time()
            execution_time = exec_end - exec_start
            stats['execution_times'].append(execution_time)

            if result.stderr:
                warnings = result.stderr.strip()
                logging.info(f"Iteration {i}: Warnings detected: {warnings}")
                stats['warnings'] += 1
                code = fix_code(code, warnings, i, recommended_model=recommended_model)
            else:
                print("Code executed successfully.", result.stdout)
                logging.info("Code executed successfully.")
                stats['successful_executions'] += 1
                beep(6)

                # Load data sample
                data_sample = get_data_sample('xdata.csv', data_sample_lines)  # 'xdata.csv' is your dataset

                # Send output to LLM with full code, dataset sample, and reasoning model's original suggestion
                llm_output_prompt = (
                    f"The code executed successfully with the following output:\n{result.stdout}\n\n"
                    f"Here is the full code that was executed:\n```python\n{code}\n```\n\n"
                    f"Here is a sample of the dataset used:\n{data_sample}\n\n"
                    f"The reasoning model ({REASONING_MODEL}) initially suggested the following:\n"
                    f"{analysis}\n\n"  # The original suggestion from the reasoning model stored earlier
                    f"Please provide additional insights or suggestions based on this information."
                )

                # Generate the response from the LLM
                llm_output_response = ollama.generate(
                    model=REASONING_MODEL,  # Use the reasoning model for analysis or insights
                    prompt=llm_output_prompt
                )

                # Optionally log or save the LLM's response
                save_raw_data(REASONING_MODEL, llm_output_prompt, llm_output_response['response'], 'output_analysis')

                if stats['successful_executions'] > 4:
                    break


        except subprocess.CalledProcessError as e:
            error_output = e.stderr.strip()
            error_output = analyze_error(error_output)
            stats['errors'] += 1
            code = fix_code(code, error_output, i, recommended_model=recommended_model)
        except Exception as e:
            logging.error(f"Unexpected error during execution: {e}\n\n")
            stats['errors'] += 1
            break

        # End timing the iteration
        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        stats['execution_times'].append(iteration_duration)

    else:
        print("Maximum iterations reached without successful execution.")
        logging.info("Maximum iterations reached without successful execution.")

    # --- Start of Reporting Statistics ---
    print("\n=== Execution Statistics ===")
    print(f"Total Iterations: {stats['total_iterations']}")
    print(f"Successful Executions: {stats['successful_executions']}")
    print(f"Warnings Encountered: {stats['warnings']}")
    print(f"Errors Encountered: {stats['errors']}")
    print(f"Code Changes Made: {stats['code_changes']}")

    if stats['execution_times']:
        average_time = sum(stats['execution_times']) / len(stats['execution_times'])
        max_time = max(stats['execution_times'])
        min_time = min(stats['execution_times'])
        print(f"Average Execution Time per Iteration: {average_time:.2f} seconds")
        print(f"Maximum Execution Time: {max_time:.2f} seconds")
        print(f"Minimum Execution Time: {min_time:.2f} seconds")
        logging.info(f"Average Execution Time per Iteration: {average_time:.2f} seconds")
        logging.info(f"Maximum Execution Time: {max_time:.2f} seconds")
        logging.info(f"Minimum Execution Time: {min_time:.2f} seconds")
    else:
        print("No execution times recorded.")

    if psutil and stats['memory_usage'] and stats['cpu_usage']:
        average_memory = sum(stats['memory_usage']) / len(stats['memory_usage'])
        max_memory = max(stats['memory_usage'])
        min_memory = min(stats['memory_usage'])
        average_cpu = sum(stats['cpu_usage']) / len(stats['cpu_usage'])
        max_cpu = max(stats['cpu_usage'])
        min_cpu = min(stats['cpu_usage'])
        print(f"Average Memory Usage: {average_memory:.2f} MB")
        print(f"Maximum Memory Usage: {max_memory:.2f} MB")
        print(f"Minimum Memory Usage: {min_memory:.2f} MB")
        print(f"Average CPU Usage: {average_cpu:.2f}%")
        print(f"Maximum CPU Usage: {max_cpu:.2f}%")
        print(f"Minimum CPU Usage: {min_cpu:.2f}%")
        logging.info(f"Average Memory Usage: {average_memory:.2f} MB")
        logging.info(f"Maximum Memory Usage: {max_memory:.2f} MB")
        logging.info(f"Minimum Memory Usage: {min_memory:.2f} MB")
        logging.info(f"Average CPU Usage: {average_cpu:.2f}%")
        logging.info(f"Maximum CPU Usage: {max_cpu:.2f}%")
        logging.info(f"Minimum CPU Usage: {min_cpu:.2f}%")

    # Export to CSV
    with open('execution_stats.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for key, value in stats.items():
            if isinstance(value, list):
                writer.writerow([key, json.dumps(value)])
            else:
                writer.writerow([key, value])

    # Export to JSON
    with open('execution_stats.json', 'w') as jsonfile:
        json.dump(stats, jsonfile, indent=4)

    # Optional: Visualize Execution Times
    if stats['execution_times']:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(stats['execution_times']) + 1), stats['execution_times'], marker='o')
        plt.title('Execution Time per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.savefig('execution_times.png')
        plt.show()
    # --- End of Reporting Statistics ---

def analyze_error(error_message):
    """
    Analyzes Python error messages, sanitizes them by excluding specific directory paths,
    and provides detailed suggestions based on the error type.

    Parameters:
    error_message (str): The full traceback or error message of the Python error.

    Returns:
    str: A sanitized and user-friendly analysis of the error.
    """
    beep(1)
    try:
        # Split the error message into lines
        tb_lines = error_message.strip().split('\n')
        
        # Initialize variables
        error_type = None
        error_detail = None
        file_name = "Unknown file"
        line_number = "Unknown line"
        function_name = "Unknown function"

        # Flag to check if traceback entries are present
        traceback_present = False

        # Extract traceback entries excluding '/site-packages/'
        tb_entries = []
        for line in tb_lines:
            if line.startswith('  File '):
                if '/site-packages/' not in line:
                    tb_entries.append(line)
                    traceback_present = True

        # If traceback entries are present, extract the last one
        if tb_entries:
            last_tb = tb_entries[-1]
            # Regex to extract file path, line number, and function name
            tb_regex = r'File "(.+?)", line (\d+), in (\w+)'
            tb_match = re.search(tb_regex, last_tb)
            if tb_match:
                full_file_path = tb_match.group(1)
                line_number = tb_match.group(2)
                function_name = tb_match.group(3)
                
                # Simplify file path by replacing the leading directories with '.../'
                sanitized_path = re.sub(r'^(.*/)([^/]+)$', r'.../\2', full_file_path)
                file_name = sanitized_path

        # Extract the error type and detail
        for line in reversed(tb_lines):
            error_match = re.match(r'^(\w+Error): (.+)', line)
            if error_match:
                error_type = error_match.group(1)
                error_detail = error_match.group(2)
                break

        if not error_type:
            # Handle cases where error_type is not found in the message
            generic_error_match = re.match(r'^(.+Error): (.+)', tb_lines[-1])
            if generic_error_match:
                error_type = generic_error_match.group(1)
                error_detail = generic_error_match.group(2)
            else:
                # If no error type can be determined
                return "Error analysis failed. Please provide a complete traceback or a detailed error message."

        if file_name == "Unknown file" or line_number == "Unknown line" or error_type is None:
                    return f"Original Error Log: {error_message}"
        
        # Define suggestions based on error type
        suggestions = {
            'SyntaxError': f"Syntax Error in {file_name} at line {line_number}: {error_detail}. "
                          "Check for missing or extra characters, incorrect indentation, or typos.",
            'ValueError': f"A ValueError occurred in {file_name} at line {line_number}: {error_detail}. "
                          "Ensure that you are passing the correct type and value of arguments.",
            'KeyError': f"A KeyError occurred in {file_name} at line {line_number}: {error_detail}. "
                        "Verify that the key exists in the dictionary before accessing it.",
            'TypeError': f"A TypeError occurred in {file_name} at line {line_number}: {error_detail}. "
                         "Check that you are using the correct data types in your operations.",
            'IndexError': f"An IndexError occurred in {file_name} at line {line_number}: {error_detail}. "
                          "Ensure that you are accessing valid indexes within the range.",
            'ImportError': f"An ImportError occurred in {file_name} at line {line_number}: {error_detail}. "
                           "Make sure the module is installed and spelled correctly.",
            'ModuleNotFoundError': f"A ModuleNotFoundError occurred in {file_name} at line {line_number}: {error_detail}. "
                                   "Install the missing module using pip, e.g., 'pip install <module_name>'.",
            'FileNotFoundError': f"A FileNotFoundError occurred in {file_name} at line {line_number}: {error_detail}. "
                                 "Ensure that the file exists and the path is correct.",
            'AttributeError': f"An AttributeError occurred in {file_name} at line {line_number}: {error_detail}. "
                              "Verify that the object has the specified attribute.",
            'ZeroDivisionError': f"A ZeroDivisionError occurred in {file_name} at line {line_number}: {error_detail}. "
                                 "Ensure that you are not dividing by zero.",
            'RuntimeError': f"A RuntimeError occurred in {file_name} at line {line_number}: {error_detail}. "
                            "Check the runtime environment and the logic of your program.",
            'NameError': f"A NameError occurred in {file_name} at line {line_number}: {error_detail}. "
                         "Ensure that the variable or function name is defined and spelled correctly.",
            'IndentationError': f"An IndentationError occurred in {file_name} at line {line_number}: {error_detail}. "
                                "Check the indentation levels in your code.",
            'NotImplementedError': f"A NotImplementedError occurred in {file_name} at line {line_number}: {error_detail}. "
                                    "Ensure that all necessary methods are implemented.",
            # Add more error types and suggestions as needed
        }

        # Construct the analysis message
        if error_type in suggestions:
            analysis = f"{error_type} in {file_name} at line {line_number}: {error_detail}\n" \
                       f"Suggestion: {suggestions[error_type]}"
        else:
            analysis = f"{error_type} in {file_name} at line {line_number}: {error_detail}\n" \
                       "Suggestion: Please check the error details and refer to the Python documentation for more information."

        return analysis
    
    except Exception as e:
        print(f"Error occurred while analyzing error: {str(e)}")
        return "Error analysis failed. Please provide a complete traceback or a detailed error message."

if __name__ == "__main__":
    main()
    beep(6)
