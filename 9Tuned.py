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

MAX_CONSECUTIVE_FAILURES = 13
MAX_SUCCESSFUL_EXECUTIONS = 13

max_unchanged_iterations = 13
iterations = 20
output_folder = 'output'
data_sample_lines = 6

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
consecutive_failures = 0

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


categorical_columns = []

target_variable = 'Exited'

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

def save_iteration_data(llm_name, prompt, response, iteration, error_output):
    llm_folder = os.path.join(raw_data_folder, llm_name)
    os.makedirs(llm_folder, exist_ok=True)

    combined_filename = os.path.join(llm_folder, f'iteration_{iteration}.txt')
    with open(combined_filename, 'w', encoding='utf-8') as combined_file:
        combined_file.write(f"Prompt:\n{prompt}\n\nError Output:\n{error_output}\n\nResponse:\n{response}")

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
        f"Analyze the data and suggest the most suitable machine learning models and plot it. use  plot show(block=False) to plot should be keep running program. "
        f"for the given dataset. Provide a one-line very short description about the choice of machine learning models. If can choose multiple models, choose. "
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

    model_match = re.search(r'is the ([A-Za-z\s]+) model', analysis)
    if model_match:
        recommended_model = model_match.group(1).strip().replace(" ", "")
    else:
        recommended_model = None
  
    if not recommended_model:
        logging.warning("No model was recommended by the LLM. Using default model selection logic.")

    code_prompt = (
        f"Based on the recommended machine learning model ({recommended_model}), write a Python program to load '{filename}', "
        f"perform preprocessing (including encoding categorical variables: {', '.join(categorical_columns)} using pd.get_dummies with appropriate parameters to avoid deprecation warnings), "
        f"split the data into training and testing sets, and apply the model. "
        f"Include code for evaluating and plotting the results, use in plot show(block=False). Do not include any comments in the code. "
        f"Actual data in file '{filename}' is not limited to sample data; the actual data file is large. Use the following data sample (10 rows) for reference:\n\n"
        f"{data_sample}*"
    )

    code_system_prompt = "Only write the code without any comments or explanations."
    reasoning_model, code_model = get_current_models()

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

def get_categorical_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_cols

def validate_schema(expected_columns, df):
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        return False, missing_columns
    return True, []

def validate_code(code, categorical_columns, target_variable):
    required_imports = [
        'from sklearn.linear_model import LogisticRegression',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve'
    ]
    for imp in required_imports:
        if imp not in code:
            return False, f"Missing import: {imp}"
    
    for col in categorical_columns:
      
        pattern = rf"pd\.get_dummies\(df, columns=\[['\"]{col}['\"]\], drop_first=True\)"
        if not re.search(pattern, code):
            return False, f"Missing or incorrect encoding for categorical column: {col}"
    
  
    target_pattern = rf"train_test_split\(.*?,\s*df\['{target_variable}'\]"
    if not re.search(target_pattern, code):
        return False, f"Incorrect target variable used. Expected '{target_variable}'."
    return True, ""

def fix_code(code, error_output, iteration, recommended_model="LogisticRegression"):
    beep(1)
    logging.info(f"Iteration {iteration}: Encountered error/warning: {error_output}")
    print(f"Iteration {iteration}: Encountered error/warning.")
    reasoning_model, code_model = get_current_models()

  
    data_sample = get_data_sample('xdata.csv', data_sample_lines)

  
    original_code_prompt = (
        f"Based on the recommended machine learning model ({recommended_model}), write a Python program to load 'xdata.csv', "
        f"perform preprocessing (including encoding categorical variables: {', '.join(categorical_columns)} using pd.get_dummies with appropriate parameters to avoid deprecation warnings), "
        f"split the data into training and testing sets, and apply the model. "
        f"Include code for evaluating and plotting the results. Do not include any comments in the code. "
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
    reasoning_model = REASONING_MODELS[current_reasoning_model_index]
    code_model = CODE_MODELS[current_code_model_index]
    logging.info(f"Using Reasoning Model: {reasoning_model}, Code Model: {code_model}")
    return reasoning_model, code_model

def switch_to_next_model():
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
    target_var = df.nunique().idxmin()
    return target_var

def main():
    global code, previous_code, unchanged_iterations, stats
    pip_executable, python_executable = create_and_activate_venv()

    consecutive_failures = 0

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

        error_output = ''

        if psutil:
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
              
                analysis = analyze_successful_execution(result.stdout, code, recommended_model)
                logging.info(f"Iteration {i}: Analysis completed.")

                if stats['successful_executions'] > MAX_SUCCESSFUL_EXECUTIONS:
                    print("Sufficient successful executions achieved. Exiting.")
                    logging.info("Sufficient successful executions achieved. Exiting.")
                    break

        except subprocess.CalledProcessError as e:
            error_output = e.stderr.strip()
            error_output = analyze_error(error_output)
            stats['errors'] += 1
            fixed_code, llm_prompt = fix_code(code, error_output, i, recommended_model=recommended_model)
            code = fixed_code

            code_model = get_current_models()[1]
            consecutive_failures += 1
            save_iteration_data(code_model, llm_prompt, fixed_code, i, error_output)

            code_model = get_current_models()[1]
            consecutive_failures += 1
            save_iteration_data(code_model, llm_prompt, fixed_code, i, error_output)

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

          
            code_model = get_current_models()[1]
            fixed_code, llm_prompt = fix_code(code, error_output, i, recommended_model=recommended_model)
            code = fixed_code
            save_iteration_data(code_model, llm_prompt, fixed_code, i, error_output)

            code_model = get_current_models()[1]
            save_iteration_data(code_model, llm_prompt, fixed_code, i, error_output)

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

def analyze_successful_execution(output, code, model):
    reasoning_model = get_reasoning_model()  
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
    # Optionally, incorporate analysis into the code
    code = integrate_analysis_into_code(code, analysis)
    return code

def extract_analysis(response_text):
    # Placeholder implementation
    return response_text

def save_stats(stats):
    import json
    with open('stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    logging.info("Statistics saved to stats.json")

def integrate_analysis_into_code(code, analysis):
    
    optimized_code = f"# Analysis: {analysis}\n" + code
    return optimized_code    

def analyze_error(error_message):
    print(f"/n/nOrignal Error:/n{error_message}/n/n")
    try:
        error_blocks = re.split(r'\*{3,}', error_message)
        analysis_results = []

        for block in error_blocks:
            block = block.strip()
            if not block:
                continue

            tb_lines = block.split('\n')
            error_type = None
            error_detail = None
            tb_entries = []
            locations = []
            problematic_line = ""
            
            for line in tb_lines:
                if line.startswith('  File '):
                    tb_entries.append(line)
                elif re.match(r'^\s*\^+', line):
                    continue 
                else:
                    error_match = re.match(r'^(\w+Error): (.+)', line)
                    if error_match:
                        error_type = error_match.group(1)
                        error_detail = error_match.group(2)

            if not error_type:
                generic_error_match = re.match(r'^(.+Error): (.+)', block, re.MULTILINE)
                if generic_error_match:
                    error_type = generic_error_match.group(1)
                    error_detail = generic_error_match.group(2)
                else:
                    analysis_results.append("Error analysis failed. Please provide a complete traceback or a detailed error message.")
                    continue

            for tb in tb_entries:
                
                tb_regex = r'File "(.+?)", line (\d+), in (.+)'
                tb_match = re.search(tb_regex, tb)
                if tb_match:
                    full_file_path = tb_match.group(1)
                    line_number = tb_match.group(2)
                
                    path_parts = full_file_path.replace("\\", "/").split('/')
                    if len(path_parts) >= 3:
                        truncated_path = '/' + '/'.join(path_parts[-3:])
                    else:
                        truncated_path = full_file_path
                    location = f"`{truncated_path}`, line {line_number}"
                    locations.append(location)

            if tb_entries:
                main_tb = tb_entries[0]
                tb_match = re.search(r'File "(.+?)", line (\d+), in (.+)', main_tb)
                if tb_match:
                    main_full_file_path = tb_match.group(1)
                    main_line_number = tb_match.group(2)
                    try:
                        with open(main_full_file_path, 'r') as f:
                            lines = f.readlines()
                            if int(main_line_number) <= len(lines):
                                problematic_line = lines[int(main_line_number) - 1].strip()
                            else:
                                problematic_line = "Problematic line not found (line number out of range)."
                    except FileNotFoundError:
                        problematic_line = "Problematic line not found (file does not exist)."
                    except Exception as e:
                        problematic_line = f"Could not read the problematic line: {str(e)}"
            else:
                problematic_line = "No traceback entries found to extract problematic line."

            if not locations or error_type is None or error_detail is None:
                analysis_results.append(f"Original Error Log: {block}")
                continue
            
            suggestions = {
                'BaseException': "A base exception occurred. Please check the traceback for more details.",
                'SystemExit': "The program is exiting. You can specify an exit status or handle the exit more gracefully.",
                'KeyboardInterrupt': "The program was interrupted by the user. Ensure that you handle interrupts appropriately.",
                'GeneratorExit': "A generator is exiting. Make sure to clean up resources properly in generator functions.",
                'Exception': "An exception occurred. Check the traceback for more details.",
                'StopIteration': "Iteration has stopped. Ensure that your iterators are functioning correctly.",
                'StopAsyncIteration': "Asynchronous iteration has stopped. Verify your asynchronous iterators.",
                'ArithmeticError': "An arithmetic error occurred. Check your mathematical operations for correctness.",
                'FloatingPointError': "A floating-point error occurred. Ensure that floating-point operations are handled correctly.",
                'OverflowError': "A numeric operation resulted in an overflow. Check your calculations for potential overflows.",
                'ZeroDivisionError': "Ensure that you are not dividing by zero in your calculations.",
                'AssertionError': "An assertion failed. Verify that your assumptions in the code are correct.",
                'AttributeError': "Verify that the object has the specified attribute.",
                'BufferError': "A buffer-related error occurred. Ensure that buffer operations are handled correctly.",
                'EOFError': "Reached end of file unexpectedly. Check file reading operations for correctness.",
                'ImportError': "Make sure the module is installed and spelled correctly.",
                'ModuleNotFoundError': "Install the missing module using pip, e.g., `pip install <module_name>`.",
                'LookupError': "A lookup operation failed. Verify the key or index you are trying to access.",
                'IndexError': "Ensure that you are accessing valid indexes within the range.",
                'KeyError': "Verify that the key exists in the dictionary before accessing it.",
                'MemoryError': "The program ran out of memory. Optimize memory usage or increase available memory.",
                'NameError': "Ensure that the variable or function name is defined and spelled correctly.",
                'UnboundLocalError': "A local variable was referenced before assignment. Check variable assignments.",
                'OSError': "An OS-related error occurred. Verify file operations and system resources.",
                'BlockingIOError': "A non-blocking operation would block. Handle blocking operations appropriately.",
                'ChildProcessError': "An operation involving a child process failed. Check child process handling.",
                'ConnectionError': "A connection error occurred. Verify network connections and configurations.",
                'BrokenPipeError': "Ensure that the pipe is open before writing to it.",
                'ConnectionAbortedError': "The connection was aborted by the peer. Check network stability.",
                'ConnectionRefusedError': "The connection attempt was refused by the peer. Verify server status.",
                'ConnectionResetError': "The connection was reset by the peer. Check network stability.",
                'FileExistsError': "Ensure that you are not trying to create a file or directory that already exists.",
                'FileNotFoundError': "Ensure that the file exists and the path is correct.",
                'InterruptedError': "The operation was interrupted. Handle interrupts appropriately.",
                'IsADirectoryError': "Ensure that you are performing file operations on a file, not a directory.",
                'NotADirectoryError': "Ensure that you are performing directory operations on a directory.",
                'PermissionError': "Ensure that you have the necessary permissions for the operation.",
                'ProcessLookupError': "The specified process does not exist. Verify the process ID.",
                'TimeoutError': "The operation timed out. Check network stability or increase timeout duration.",
                'ReferenceError': "A weak reference proxy was used to access a garbage collected referent.",
                'RuntimeError': "A runtime error occurred. Check the logic of your program.",
                'NotImplementedError': "Ensure that all necessary methods are implemented.",
                'RecursionError': "The maximum recursion depth was exceeded. Optimize recursive calls.",
                'SyntaxError': "Check for syntax issues like missing or extra characters, incorrect indentation, or typos.",
                'IndentationError': "Check the indentation levels in your code.",
                'TabError': "Inconsistent use of tabs and spaces in indentation. Ensure consistent indentation.",
                'SystemError': "A system error occurred. Check internal Python errors or report a bug.",
                'TypeError': "Check that you are using the correct data types in operations.",
                'ValueError': "Ensure that the values are of the correct type and within valid ranges.",
                'UnicodeError': "A Unicode-related error occurred. Verify encoding and decoding operations.",
                'UnicodeDecodeError': "Ensure that you are decoding using the correct encoding.",
                'UnicodeEncodeError': "Ensure that you are encoding using the correct encoding.",
                'UnicodeTranslateError': "Ensure that you are translating using the correct encoding.",
                'Warning': "A warning was issued. Check the code for potential issues.",
                'DeprecationWarning': "A deprecated feature was used. Update the code to use current features.",
                'PendingDeprecationWarning': "A feature is pending deprecation. Prepare to update the code.",
                'RuntimeWarning': "A runtime warning occurred. Check the code for potential issues.",
                'SyntaxWarning': "A syntax warning occurred. Check the code for potential syntax issues.",
                'UserWarning': "A user-generated warning was issued. Review the code for potential issues.",
                'FutureWarning': "A feature will change in the future. Update the code to be compatible with future versions.",
                'ImportWarning': "An import-related warning occurred. Check module imports for potential issues.",
                'UnicodeWarning': "A Unicode-related warning occurred. Verify encoding and decoding operations.",
                'BytesWarning': "A bytes-related warning occurred. Check operations involving bytes.",
                'ResourceWarning': "A resource was not properly released. Ensure that resources are managed correctly.",
            }

            if error_type in suggestions:
                suggestion = suggestions[error_type]
            else:
                suggestion = "Please check the error details and refer to the Python documentation for more information."

            analysis = (f"**Error Type:** {error_type}\n"
                        f"**Location:**\n" +
                        "\n".join([f"- {loc}" for loc in locations]) + "\n"
                        f"**Error Detail:** {error_detail}\n"
                        f"**Problematic Line:** `{problematic_line}`\n"
                        f"**Suggestion:** {suggestion}")

            analysis_results.append(analysis)

        final_analysis = "\n\n".join(analysis_results)
        return final_analysis

    except Exception as e:
        
        print(f"Error occurred while analyzing error: {str(e)}")
        return "Error analysis failed. Please provide a complete traceback or a detailed error message."


if __name__ == "__main__":
    main()
    beep(6)
