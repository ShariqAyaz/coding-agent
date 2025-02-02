import os
import re
import ollama
import subprocess

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

iterations = 30
code = ''  # Initial code

def extract_code(llm_output):
    # Remove code fences and any non-code text
    code_pattern = r'```(?:python)?\n(.*?)```'
    code_matches = re.findall(code_pattern, llm_output, re.DOTALL)
    if code_matches:
        code = '\n'.join(code_matches)
    else:
        # Remove any markdown or explanations
        lines = llm_output.split('\n')
        code_lines = []
        code_block_started = False
        for line in lines:
            # Start collecting code after ```python or ``` line
            if line.strip().startswith('```'):
                code_block_started = not code_block_started
                continue
            if code_block_started:
                code_lines.append(line)
            else:
                # Try to collect code outside code blocks if it looks like code
                if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('*'):
                    code_lines.append(line)
        code = '\n'.join(code_lines)
    # Further clean the code by removing any markdown artifacts
    code = re.sub(r'```', '', code)
    return code.strip()

def clean_code(code):
    """
    Clean redundant imports and repeated lines of code.
    """
    # Remove repeated import statements
    cleaned_lines = []
    seen_imports = set()

    for line in code.split('\n'):
        # If it's an import statement, track it and avoid duplicates
        if line.strip().startswith('import') or line.strip().startswith('from'):
            if line not in seen_imports:
                seen_imports.add(line)
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    # Join cleaned lines back together
    cleaned_code = '\n'.join(cleaned_lines)
    
    # Remove any extra redundant empty lines or repeated data-loading
    cleaned_code = re.sub(r'\n{2,}', '\n', cleaned_code)  # Limit consecutive newlines to 1
    return cleaned_code.strip()

def generate_initial_code():
    global code
    if not code:
        # Generate initial code via LLM
        prompt = "Write a Python program to load a CSV (Salary_Data.csv), analyze its data for what regression, perform preprocessing, train-test split, and plot the results."
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
    # Extract line number from the error message
    pattern = r'File ".*", line (\d+)(.*)'
    matches = re.findall(pattern, error_output)
    module_not_found_pattern = r"No module named '([^']+)'"
    module_match = re.search(module_not_found_pattern, error_output)
    if module_match:
        missing_module = module_match.group(1)
        print(f"Missing module '{missing_module}'. Attempting to install.")
        subprocess.run(['pip', 'install', missing_module])
        return code
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
                # Fix indentation if necessary
                indent = re.match(r'^(\s*)', faulty_line).group(1)
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
    global code
    generate_initial_code()
    if not code:
        print("No code was generated. Exiting.")
        return
    for i in range(iterations):
        print(f"\nIteration {i+1}:")
        code = clean_code(code)  # Clean code before running
        file_path = os.path.join(output_folder, f'generated_code_{i}.py')
        with open(file_path, 'w') as f:
            f.write(code)
        try:
            print("Running the code...")
            result = subprocess.run(['python', file_path], check=True, capture_output=True, text=True)
            print("Code ran successfully.")
            print("Program output:")
            print(result.stdout)
            break
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.strip()
            print("Code execution failed with error:")
            print(error_output)
            code = fix_code(code, error_output)
    else:
        print(f"Failed to execute code after {iterations} iterations.")
    print("\nFinal code:")
    print(code)

if __name__ == "__main__":
    main()
