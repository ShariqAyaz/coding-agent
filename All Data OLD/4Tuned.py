import os
import re
import ollama
import subprocess
import sys
import pandas as pd

output_folder = 'output'
raw_data_folder = 'raw-data'
venv_dir = os.path.join(output_folder, 'venv')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(raw_data_folder, exist_ok=True)

iterations = 30
code = ''
previous_code = None  
unchanged_iterations = 0  
max_unchanged_iterations = 5  

data_sample_lines = 10  

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

def get_data_sample(file_path, num_lines=10):
    try:
        df = pd.read_csv(file_path, nrows=num_lines)
        return df.to_csv(index=False)
    except Exception as e:
        return ""

def save_raw_data(llm_name, prompt, response, iteration):
    llm_folder = os.path.join(raw_data_folder, llm_name)
    os.makedirs(llm_folder, exist_ok=True)
    with open(os.path.join(llm_folder, f'prompt_{iteration}.txt'), 'w', encoding='utf-8') as f:
        f.write(prompt)
    with open(os.path.join(llm_folder, f'response_{iteration}.txt'), 'w', encoding='utf-8') as f:
        f.write(response)

def generate_initial_code():
    global code
    data_sample = get_data_sample('sdata.csv', data_sample_lines)
    if not data_sample:
        sys.exit(1)
    
    reasoning_prompt = (
        f"Given the following data sample from 'sdata.csv':\n\n{data_sample}\n\n"
        f"Analyze the data and suggest the most suitable machine learning model to use "
        f"for this dataset. Provide a brief reasoning for your choice."
    )
    reasoning_system_prompt = "Analyze the data and recommend the best ML model."
    reasoning_model = 'llama3.1'

    reasoning_response = ollama.generate(
        model=reasoning_model,
        prompt=reasoning_prompt,
        system=reasoning_system_prompt
    )
    analysis = reasoning_response['response'].strip()
  
    save_raw_data(reasoning_model, reasoning_prompt, analysis, 'initial')
    
    code_prompt = (
        f"Based on the recommended machine learning model, write a Python program to load 'sdata.csv', "
        f"perform preprocessing (including encoding categorical variables), split the data into training and testing sets, and apply the model. "
        f"Include code for evaluating and plotting the results. Do not include any comments in the code. "
        f"Actual data is not limited to sample data; the actual data file is large. Use the following data sample (10 rows) for reference:\n\n"
        f"{data_sample}"
    )
    code_system_prompt = "Only write the code without any comments or explanations."
    code_model = 'codellama:13b'

    code_response = ollama.generate(
        model=code_model,
        prompt=code_prompt,
        system=code_system_prompt
    )
    llm_output = code_response['response'].strip()

    save_raw_data(code_model, code_prompt, llm_output, 'initial')

    code = extract_code(llm_output)

def fix_code(code, error_output, iteration):
    print(f"Iteration {iteration}")
    
    # Pattern to detect missing modules
    module_not_found_pattern = r"No module named ['\"]([^'\"]+)['\"]"
    module_match = re.search(module_not_found_pattern, error_output)
    
    if module_match:
        missing_module = module_match.group(1)
        pip_executable, _ = create_and_activate_venv()
        if missing_module == 'sklearn':
            subprocess.run([pip_executable, 'install', 'scikit-learn'], check=True)
        else:
            subprocess.run([pip_executable, 'install', missing_module], check=True)
        return code

    # For other errors, provide more context
    llm_prompt = (
        f"The following Python code is producing an error:\n\n{code}\n\n"
        f"Error message:\n{error_output}\n\n"
        f"The dataset includes categorical variables that may need encoding. Please fix the code to handle these issues. Only provide the corrected code without any explanations or comments."
    )
    system_prompt = "Only write the corrected code without any comments or explanations."
    llm_model = 'codellama:13b'

    response = ollama.generate(
        model=llm_model,
        prompt=llm_prompt,
        system=system_prompt
    )
    fixed_code = response['response'].strip()

    save_raw_data(llm_model, llm_prompt, fixed_code, iteration)

    code = extract_code(fixed_code)
    return code

def main():
    global code, previous_code, unchanged_iterations
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

    generate_initial_code()

    if not code:
        return

    for i in range(iterations):
        print('Iteration:', i)
        file_path = os.path.join(output_folder, f'generated_code_{i}.py')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        if previous_code == code:
            unchanged_iterations += 1
        else:
            unchanged_iterations = 0
        previous_code = code

        if unchanged_iterations >= max_unchanged_iterations:
            print("No improvements detected. Stopping iterations.")
            break

        try:
            result = subprocess.run([python_executable, file_path], check=True, capture_output=True, text=True)
            print("Code executed successfully.")
            break
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.strip()
            code = fix_code(code, error_output, i)
        except Exception as e:
            break
    else:
        print("Maximum iterations reached without successful execution.")

if __name__ == "__main__":
    main()
