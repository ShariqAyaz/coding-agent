import os
import re
import ollama
import subprocess
import sys
import pandas as pd

output_folder = 'output'
venv_dir = os.path.join(output_folder, 'venv')

# Ensure the 'output' folder exists
os.makedirs(output_folder, exist_ok=True)

iterations = 30
code = ''
previous_code = None  # Track the previous iteration's code
unchanged_iterations = 0  # Track how many iterations had unchanged code
max_unchanged_iterations = 3  # Max unchanged iterations before breaking

data_sample_lines = 10  # Number of lines to sample from the dataset

def create_and_activate_venv():
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in '{venv_dir}'...")
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)

    # Determine paths for pip and python inside the virtual environment
    if os.name == 'nt':  # Windows
        pip_executable = os.path.join(venv_dir, 'Scripts', 'pip.exe')
        python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe')
    else:  # Unix-based (Linux, macOS)
        pip_executable = os.path.join(venv_dir, 'bin', 'pip')
        python_executable = os.path.join(venv_dir, 'bin', 'python')

    return pip_executable, python_executable

def extract_code(llm_output):
    # Remove code fences and any non-code text
    code_pattern = r'```(?:python)?\n(.*?)```'
    code_matches = re.findall(code_pattern, llm_output, re.DOTALL)
    if code_matches:
        code = '\n'.join(code_matches)
    else:
        code = llm_output.strip()
    # Further clean the code by removing any markdown artifacts
    code = re.sub(r'```', '', code)
    return code.strip()

def get_data_sample(file_path, num_lines=10):
    try:
        df = pd.read_csv(file_path, nrows=num_lines)
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Error reading data sample: {e}")
        return ""

def generate_initial_code():
    global code
    data_sample = get_data_sample('sdata.csv', data_sample_lines)
    if not data_sample:
        print("Unable to read data sample. Exiting.")
        sys.exit(1)

    # Use llama3.1 for reasoning about the dataset
    reasoning_prompt = (
        f"Given the following data sample from 'sdata.csv':\n\n{data_sample}\n\n"
        f"Analyze the data and suggest the most suitable machine learning model to use "
        f"for this dataset. Provide a brief reasoning for your choice."
    )
    print("Generating analysis using LLM (llama3.1)...")
    reasoning_response = ollama.generate(
        model='llama3.1',
        prompt=reasoning_prompt,
        system="Analyze the data and recommend the best ML model."
    )
    analysis = reasoning_response['response'].strip()
    print("LLM analysis:")
    print(analysis)

    # Now generate the initial code using a code generation model
    code_prompt = (
        f"Based on the recommended machine learning model, write a Python program to load 'sdata.csv', "
        f"perform necessary preprocessing, split the data into training and testing sets, and apply the model. "
        f"Include code for evaluating and plotting the results. Use the following data sample for reference:\n\n"
        f"{data_sample}"
    )
    system_prompt = "No opening or closing statements, no instructions, no comments, no explanations. Only write the code."
    print("Generating initial code using code generation model...")
    response = ollama.generate(
        model='codellama:13b',
        prompt=code_prompt,
        system=system_prompt
    )
    llm_output = response['response'].strip()
    print("LLM output:")
    print(llm_output)
    code = extract_code(llm_output)
    print("Initial code extracted:")
    print(code)

def fix_code(code, error_output):
    print("Fixing code using LLM...")
    print("Error output:")
    print(error_output)

    # Check for ModuleNotFoundError
    module_not_found_pattern = r"No module named ['\"]([^'\"]+)['\"]"
    module_match = re.search(module_not_found_pattern, error_output)

    if module_match:
        missing_module = module_match.group(1)
        print(f"Missing module '{missing_module}'. Attempting to install.")
        pip_executable, _ = create_and_activate_venv()
        # Special handling for common module name differences
        if missing_module == 'sklearn':
            subprocess.run([pip_executable, 'install', 'scikit-learn'], check=True)
        else:
            subprocess.run([pip_executable, 'install', missing_module], check=True)
        return code

    # Provide the entire code and error message to the LLM
    llm_prompt = (
        f"The following Python code is producing an error:\n\n{code}\n\n"
        f"Error message:\n{error_output}\n\n"
        f"Please fix the code. Only provide the corrected code without any explanations."
    )
    system_prompt = "No opening or closing statements, no instructions, no comments, no explanations. Only write the code."
    print("LLM fix prompt:")
    print(llm_prompt)
    response = ollama.generate(
        model='codellama:13b',
        prompt=llm_prompt,
        system=system_prompt
    )
    fixed_code = response['response'].strip()
    code = extract_code(fixed_code)
    print("Updated code:")
    print(code)
    return code

def main():
    global code, previous_code, unchanged_iterations
    pip_executable, python_executable = create_and_activate_venv()

    # Activate the virtual environment and install necessary packages
    print("Activating virtual environment and installing dependencies...")
    subprocess.run([pip_executable, 'install', '--upgrade', 'pip'], check=True)
    # Initial dependencies, adjust as needed
    initial_dependencies = [
        'pandas',
        'matplotlib',
        'scikit-learn',
        'numpy'
    ]
    subprocess.run([pip_executable, 'install'] + initial_dependencies, check=True)

    generate_initial_code()

    if not code:
        print("No code was generated. Exiting.")
        return

    for i in range(iterations):
        print(f"\nIteration {i+1}:")
        file_path = os.path.join(output_folder, f'generated_code_{i}.py')  # Ensure files are saved inside output folder
        with open(file_path, 'w') as f:
            f.write(code)

        # Check if the code has changed
        if previous_code == code:
            unchanged_iterations += 1
            print(f"Code unchanged for {unchanged_iterations} consecutive iterations.")
        else:
            unchanged_iterations = 0
        previous_code = code

        # If code remains unchanged for too many iterations, break early
        if unchanged_iterations >= max_unchanged_iterations:
            print(f"Code has remained unchanged for {max_unchanged_iterations} iterations. Exiting.")
            break

        try:
            print("Running the code...")
            result = subprocess.run([python_executable, file_path], check=True, capture_output=True, text=True)
            print("Code ran successfully.")
            print("Program output:")
            print(result.stdout)
            break
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.strip()
            print("Code execution failed with error:")
            print(error_output)
            code = fix_code(code, error_output)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    else:
        print(f"Failed to execute code after {iterations} iterations.")

    print("\nFinal code:")
    print(code)

if __name__ == "__main__":
    main()
