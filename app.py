import os
import shutil
import json
import re
import subprocess
import sys
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile
from typing import List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok

from model import model
import os
import re
import json
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # allow requests from any origin
    allow_credentials=True,         # allow cookies, authorization headers, etc.
    allow_methods=["*"],            # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # allow all headers
)



def extract_questions(description: str, attachments: list[str] = None) -> tuple[list[str], str]:
    """
    Extract questions and output format by:
    1. Looking for 'questions.txt' or 'question.txt' in attachments under 'extracted' folder.
    2. Reading its contents and parsing questions locally (one per line).
    3. Calling LLM only if needed (for output_format detection or tricky cases).
    Returns: (questions, output_format)
    """
    attachments = attachments or []
    questions_txt_content = ""
    questions = []
    output_format = ""
    import os

    extracted_dir = "extracted"  

    for file_path in attachments:
     
        extracted_path = os.path.join(extracted_dir, os.path.basename(file_path))
        normalized_path = os.path.normpath(extracted_path).lower()

        print(f"Checking in extracted folder: {extracted_path}")
        print(f"Normalized: {normalized_path}")

        if normalized_path.endswith("questions.txt") or \
        normalized_path.endswith("question.txt"):

            print("finally here")
            try:
                with open(extracted_path, "r", encoding="utf-8") as f:
                    questions_txt_content = f.read().strip()
                    print(questions_txt_content)
                print(f"[✓] Loaded questions.txt from: {extracted_path}")
            except Exception as e:
                print(f"[!] Failed to read {extracted_path}: {e}")
            break

    if questions_txt_content:
     
        questions = [line.strip() for line in questions_txt_content.splitlines() if line.strip()]
        print(f"[✓] Found {len(questions)} questions locally.")


    if not output_format or not questions:
        try:
            prompt = f"""
You are given:
1. A task description: {description}
2. The content of a questions.txt file:

\"\"\"{questions_txt_content or '[No questions.txt provided]'}\"\"\"

Instructions:
- List all questions from the file or description in the "questions" array.
- If questions are already clear, repeat them exactly as they appear without any extra strings or space or anything.
- Determine the required output format from the description or file content. 
- make sure your number of question matches the number of questions mentioned in the file.

  If not specified, use an empty string for output_format.

Return only valid JSON in this format:
{{
  "questions": ["Question 1", "Question 2", ...],
  "output_format": "Expected output format or empty string"
}}
"""
            messages = [
                SystemMessage("You are an AI that extracts questions and output formats from text."),
                HumanMessage(prompt)
            ]
            response = model.invoke(messages)
            print(f"response : {response}")

            try:
                data = json.loads(response.content)
            except Exception:
                print("[!] Could not parse JSON directly, attempting regex match...")
                json_match = re.search(r"\{.*\}", response.content, re.S)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in LLM response.")

            if not questions:
                questions = data.get("questions", [])
            output_format = data.get("output_format", "")

            print(f"[✓] Extracted {len(questions)} questions and output format from LLM.")
        except Exception as e:
            print(f"[!] LLM extraction failed: {e}")

    return questions, output_format



def execute_code(code: str, language: str, working_dir: str) -> dict:
    """Execute code and return results"""
    try:
        if language.lower() == "python":
          
            code_file = os.path.join(working_dir, "generated_code.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            
        
            result = subprocess.run([sys.executable, code_file], 
                                  capture_output=True, text=True, timeout=60,
                                  cwd=working_dir)
            
            return {
                "executed": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                  "return_code": None,
                "language": "Python",
                "success": result.returncode == 0
            }
            
        elif language.lower() == "javascript":
       
            code_file = os.path.join(working_dir, "generated_code.js")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            
     
            result = subprocess.run(['node', code_file], 
                                  capture_output=True, text=True, timeout=60,
                                  cwd=working_dir)
            
            return {
                "executed": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                       "return_code": None,
                "language": "JavaScript",
                "success": result.returncode == 0
            }
        
        else:
            return {
                "executed": False,
                "error": f"Unsupported language: {language}",
                "language": language,
                "success": False
            }
            
    except subprocess.TimeoutExpired:
        return {
            "executed": False,
            "error": "Code execution timeout (60 seconds)",
            "language": language,
            "success": False
        }
    except Exception as e:
        return {
            "executed": False,
            "error": str(e),
            "language": language,
            "success": False
        }


def generate_and_execute_code_for_questions(file_paths: list[str], questions: list[str], output_format: str, description: str, max_retries: int = 5) -> dict:
    """
    Generate code to answer questions, execute it, and return results with retry logic
    """
    
    def read_file_content(file_path: str) -> str:
        """Read file content based on file type"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.xml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                file_size = os.path.getsize(file_path)
                return f"[Binary file: {os.path.basename(file_path)}, Size: {file_size} bytes]"
        except Exception as e:
            return f"[Error reading {file_path}: {str(e)}]"
    
    def create_code_generation_prompt(files_content: dict, questions: list[str], output_format: str, description: str, error_context: str = "") -> str:
        """Create prompt for LLM to generate code"""
        
        files_section = "\n\n".join([
            f"=== FILE: {filename} ===\n{content}" 
            for filename, content in files_content.items()
        ])
        
        questions_section = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        error_section = f"\n\nPREVIOUS ERROR CONTEXT:\n{error_context}" if error_context else ""
        
        prompt = f"""
TASK DESCRIPTION: {description}

FILES PROVIDED:
{files_section}

QUESTIONS TO ANSWER:
{questions_section}

EXPECTED OUTPUT FORMAT: {output_format}
{error_section}

Please analyze the provided files and generate Python code that will:
1. Process/analyze the data from the files
2. Answer all the questions listed above
3. Return the results in the exact format specified
4. dont use dummy data for analysis
5. do necessary data cleaning as per the questions.
6. do check the format of the data before generating the code.
7. do check the dtype of the columns before generating the code.
# Add this to the prompt where you have "CRITICAL: For JSON output..."
8. IMPORTANT: Convert numpy/pandas types before JSON serialization:
   - Use int(value) for numpy integers (int64, int32, etc.)
   - Use float(value) for numpy floats  
   - Use value.item() for numpy scalars
Your response should contain:
1. The complete Python code that can be executed
2. The code should handle file reading, data processing, and output formatting
3. Include proper error handling in your code
4. Print the final results in the expected format

Return your response in this exact format:
```python
# Your complete Python code here
```

If this is a retry, please fix the previous error and provide corrected code.
"""
        return prompt
    
   
    retry_count = 0
    last_error = ""
    files_content = {}
    working_dir = os.path.dirname(file_paths[0]) if file_paths else "extracted"
    working_dir = os.path.abspath("extracted")

    
 
    print(f"[✓] Reading {len(file_paths)} files for code generation...")
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        content = read_file_content(file_path)
        files_content[filename] = content
        print(f"[✓] Read file: {filename}")
    

    while retry_count < max_retries:
        try:
            print(f"[→] Code generation attempt {retry_count + 1}/{max_retries}")
            

            error_context = f"Attempt {retry_count + 1}. Previous error: {last_error}" if retry_count > 0 else ""
            prompt = create_code_generation_prompt(files_content, questions, output_format, description, error_context)
            

            messages = [
                SystemMessage("You are an expert Python developer and data analyst. Generate executable Python code to solve the given tasks."),
                HumanMessage(prompt)
            ]
        
            response = model.invoke(messages)
            response_content = response.content
            

            code_match = re.search(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
            if not code_match:
   
                code_match = re.search(r'```\s*(.*?)\s*```', response_content, re.DOTALL)
            
            if not code_match:
                raise ValueError("No Python code found in LLM response")
            
            generated_code = code_match.group(1).strip()
            print(f"[✓] Generated code ({len(generated_code)} characters)")
            
          
            print(f"[→] Executing generated code...")
            execution_result = execute_code(generated_code, "python", working_dir)
            
            if execution_result["success"]:
                print(f"[✓] Code executed successfully on attempt {retry_count + 1}")
                
                return {
                    "success": True,
                    "results": execution_result["stdout"],
                    "generated_code": generated_code,
                    "execution_details": execution_result,
                    "attempts": retry_count + 1,
                    "files_processed": list(files_content.keys()),
                    "questions_answered": len(questions),
                    "errors": []
                }
            else:
 
                error_msg = f"Code execution failed. Return code: {execution_result}"
                if execution_result.get("stderr"):
                    error_msg += f"\nStderr: {execution_result['stderr']}"
                if execution_result.get("error"):
                    error_msg += f"\nError: {execution_result['error']}"
                
                raise ValueError(error_msg)
            
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            print(f"[!] Attempt {retry_count} failed: {last_error}")
            
            if retry_count >= max_retries:
                print(f"[✗] Maximum retries ({max_retries}) reached. Code generation/execution failed.")
                break
            else:
                print(f"[→] Retrying... ({retry_count + 1}/{max_retries})")
    
    return {
        "success": False,
        "results": None,
        "attempts": retry_count,
        "files_processed": list(files_content.keys()),
        "questions_answered": 0,
        "errors": [last_error],
        "final_attempt": retry_count,
        "max_retries_reached": True
    }


@app.post("/analyze0")
async def analyze_data0(request: Request):
    description = "Analyze the data and answer the questions."
    
    extracted_folder = "extracted"
    


    form = await request.form()
    print(f"Form data received: {form}")

    file_names = []
    file_paths = []
    extracted_folder = "extracted"
    os.makedirs(extracted_folder, exist_ok=True)

    for form_key in form.keys():
        upload_file = form[form_key]
        from starlette.datastructures import UploadFile

        if isinstance(upload_file, UploadFile) and  upload_file.filename:
            print(f"Processing upload_file: {upload_file} (type: {type(upload_file)})")
            if isinstance(upload_file, UploadFile) and upload_file.filename:
                safe_name = upload_file.filename.strip().replace(" ", "_")
                file_path = os.path.join(extracted_folder, safe_name)

                content = await upload_file.read()
                print(f"Read {len(content)} bytes from {safe_name}")

                with open(file_path, "wb") as f:
                    f.write(content)
                print(f"Saved file to {file_path}")

                file_names.append(safe_name)
                file_paths.append(file_path)
            else:
                print(f"Skipped file: {upload_file}")

    all_files_in_folder = []
    for filename in os.listdir(extracted_folder):
        if os.path.isfile(os.path.join(extracted_folder, filename)):
            all_files_in_folder.append(filename)

    print(f"📂 Uploaded Files recorded in list: {file_names}")
    print(f"📂 Files physically in extracted folder: {all_files_in_folder}")


    questions, output_format = extract_questions(description, all_files_in_folder)
    print(questions, output_format)
    print('see')

    from fastapi import UploadFile  
    
    if questions and len(questions) > 0:
        print(f"[✓] Starting code generation and execution with {len(questions)} questions...")
        analysis_results = generate_and_execute_code_for_questions(
            file_paths=file_paths,
            questions=questions,
            output_format=output_format,
            description=description,
            max_retries=5
        )
        print(f"[✓] Code generation/execution completed. Success: {analysis_results.get('success', False)}")
    else:
        print("[!] No questions found, skipping analysis")
        analysis_results = {
            "success": False,
            "message": "No questions to analyze",
            "files_processed": all_files_in_folder
        }
    
    print(analysis_results)
    print(type(analysis_results))
    if analysis_results.get("success", False):
        try:
   
            output_dict = json.loads(analysis_results["results"])
            print(f' afterwards: {type(output_dict)}')
            return output_dict
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse results as JSON",
                "raw_results": analysis_results["results"],
                "parse_error": str(e)
            }
    else:
        return {
            "error": "Analysis failed after maximum retries",
            "details": analysis_results.get("errors", ["Unknown error"]),
            "attempts": analysis_results.get("attempts", 0)
        }

if __name__ == "__main__":
    from pyngrok import ngrok

    public_url = ngrok.connect(
        addr=8000,
        proto="http",
        domain="domain.ngrok-free.app" 
    )
    print(f"Public Ngrok URL: {public_url}")

    # Start uvicorn programmatically
    uvicorn.run("x:app", host="127.0.0.1", port=8000, reload=True,    workers=1,  # Keep as 1 with reload=True
    limit_concurrency=5,  # This increases concurrency
    access_log=False,  # Reduces log noise
        reload_excludes=["extracted/*", "*.tmp", "generated_code.py"],  # Exclude the extracted folder # Avoid reload warnings
)
