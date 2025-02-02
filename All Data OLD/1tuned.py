import os
import re
import ollama
import subprocess
import sys

output_folder = 'output'
venv_dir = os.path.join(output_folder, 'venv')

# Make sure the 'output' folder exists
os.makedirs(output_folder, exist_ok=True)

iterations = 30
code = ''
previous_code = None  # Variable to track the previous iteration's code
unchanged_iterations = 0  # To track how many iterations had unchanged code
max_unchanged_iterations = 3  # Maximum unchanged iterations before breaking

def create_and_activate_venv():
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in '{output_folder}' folder...")
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)

    # Determine paths for pip and python inside the virtual environment
    if os.name == 'nt':  # Windows
        pip_executable = os.path.join(venv_dir, 'Scripts', 'pip')
        python_executable = os.path.join(venv_dir, 'Scripts', 'python')
    else:  # Unix-based (Linux, macOS)
        pip_executable = os.path.join(venv_dir, 'bin', 'pip')
        python_executable = os.path.join(venv_dir, 'bin', 'python')
    
    return pip_executable, python_executable

def extract_code(llm_output):
    code_pattern = r'```(?:python)?\n(.*?)```'
    code_matches = re.findall(code_pattern, llm_output, re.DOTALL)
    if code_matches:
        code = '\n'.join(code_matches)
    else:
        lines = llm_output.split('\n')
        code_lines = []
        code_block_started = False
        for line in lines:
            if line.strip().startswith('```'):
                code_block_started = not code_block_started
                continue
            if code_block_started:
                code_lines.append(line)
            else:
                if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('*'):
                    code_lines.append(line)
        code = '\n'.join(code_lines)

    code = re.sub(r'```', '', code)
    return code.strip()

def generate_initial_code():
    global code
    if not code:
        prompt = "Write a Python program to load a CSV (sdata.csv), analyze its data using any machine learning models, perform preprocessing, train-test split, and plot the results."
        system_prompt = "No opening or closing statements, no instructions, no comments, no explanations. Only write the code."
        print("Generating initial code using LLM...")
        response = ollama.generate(model='deepseek-coder-v2', prompt=prompt, system=system_prompt)
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
  
    pattern = r'File ".*", line (\d+)(.*)'
    matches = re.findall(pattern, error_output)
    module_not_found_pattern = r"No module named ['\"]([^'\"]+)['\"]"
    module_match = re.search(module_not_found_pattern, error_output)
    
    # If module not found, install it using the venv pip
    if module_match:
        missing_module = module_match.group(1)
        print(f"Missing module '{missing_module}'. Attempting to install.")
        pip_executable, _ = create_and_activate_venv()
        subprocess.run([pip_executable, 'install', missing_module], check=True)
        return code
    
    # Handle other code errors
    if matches:
        for match in matches:
            line_number = int(match[0])
            error_message = match[1].strip()
            code_lines = code.split('\n')
            if 0 <= line_number - 1 < len(code_lines):
                faulty_line = code_lines[line_number - 1]
                print(f"Faulty line {line_number}: {faulty_line}")
                llm_prompt = (
                    f"Fix the following line in Python code:\n{faulty_line}\n"
                    f"Error: {error_message}\n"
                    f"Only provide the fixed line of code, without any additional text."
                )
                system_prompt = "No opening or closing statements, no instructions, no comments, no explanations. Only write the code."
                print("LLM fix prompt:")
                print(llm_prompt)
                response = ollama.generate(model='deepseek-coder-v2', prompt=llm_prompt, system=system_prompt)
                fixed_line = response['response'].strip()
                fixed_line = extract_code(fixed_line)
                print("LLM fixed line:")
                print(fixed_line)
              
                # Preserve indentation
                indent_match = re.match(r'^(\s*)', faulty_line)
                indent = indent_match.group(1) if indent_match else ''
                fixed_line = indent + fixed_line.strip()
                code_lines[line_number - 1] = fixed_line
                code = '\n'.join(code_lines)
    else:
        llm_prompt = (
            f"Please fix the following Python code:\n\n{code}\n\nError:\n{error_output}\n"
            f"Only provide the corrected code without any explanations."
        )
        system_prompt = "No opening or closing statements, no instructions, no comments, no explanations. Only write the code."
        print("LLM fix prompt:")
        print(llm_prompt)
        response = ollama.generate(model='deepseek-coder-v2', prompt=llm_prompt, system=system_prompt)
        code = response['response'].strip()
        code = extract_code(code)
        print("Updated code:")
        print(code)
    return code

def main():
    global code, previous_code, unchanged_iterations
    pip_executable, python_executable = create_and_activate_venv()
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
