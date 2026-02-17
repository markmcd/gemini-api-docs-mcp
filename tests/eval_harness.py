import asyncio
import json
import os
import subprocess
import sys
import shutil
import argparse
import re
import datetime
from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict, Any, List, Optional

# Configuration
DEFAULT_MODEL = "gemini-3-flash-preview"
PROMPTS_FILE = "tests/test_prompts.json"
GENERATED_DIR = "tests/generated"
RESULT_FILE = "tests/result.json"
SKILL_PATH = "skills/gemini-api-dev"

# Model validation patterns
DEPRECATED_MODELS = {
    'gemini-1.0',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'gemini-2.0-flash',
    'gemini-2.0-pro',
}

CURRENT_MODELS = {
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-3-pro-preview',
    'gemini-3-flash-preview',
    'gemini-3-pro-image-preview',
}

server_params = StdioServerParameters(
    command="python3",  # Executable
    args=["-m", "gemini_docs_mcp.server"],  # MCP Server
    env=None,  # Optional environment variables
)

# --- Code Extraction & Analysis Utils ---

def extract_code_py(response_str: str) -> str:
    """Extracts code for the given language from the response."""
    re_pattern = rf'```python\n.*?\n\s*```'
    compiled_pattern = re.compile(re_pattern, re.DOTALL)

    if found := compiled_pattern.findall(response_str):
        found = [s.strip() for s in found]
        if len(found) == 1:
            return found[0].replace("```python", "").replace("```", "").strip()

        result_str_list = []
        for i, s in enumerate(found):
            result_str_list.append(f'# chunk {i+1}')
            result_str_list.append(s.replace("```python", "").replace("```", "").strip())
        return '\n'.join(result_str_list)
    else:
        return response_str.strip()

def dedent_code_str(code_str: str) -> str:
    """Dedents the given code string."""
    code_str_lines = code_str.split('\n')
    if not code_str_lines: return code_str
    # Find first non-empty line to determine indentation
    first_line = next((line for line in code_str_lines if line.strip()), None)
    if not first_line: return code_str
    
    n_dedent = len(first_line) - len(first_line.lstrip())
    if n_dedent > 0:
        return '\n'.join([line[n_dedent:] if len(line) >= n_dedent else line for line in code_str_lines])
    return code_str

def extract_code_ts(response_str: str) -> str:
    """Extracts code for the given language from the response."""
    re_pattern = rf'```(?:typescript|javascript)\n.*?\n\s*```'
    compiled_pattern = re.compile(re_pattern, re.DOTALL)

    if found := compiled_pattern.findall(response_str):
        found = [s.strip() for s in found]

        result_str_list = []
        for i, s in enumerate(found):
            # check language
            if s.startswith('```typescript'):
                lang = 'typescript'
            elif s.startswith('```javascript'):
                lang = 'javascript'
            else:
                lang = 'unknown'
            result_str_list.append(f'//{lang} chunk {i+1}')

            # strip the quotes and dedent
            code_chunk = '\n'.join(s.split('\n')[1:-1])
            s = dedent_code_str(code_chunk)
            result_str_list.append(s)
        return '\n'.join(result_str_list)
    else:
        return '// unable to extract code\n' + response_str

def extract_code(response_str: str, lang: str) -> str:
    if lang == 'python':
        return extract_code_py(response_str)
    elif lang == 'typescript':
        return extract_code_ts(response_str)
    else:
        # Fallback for unspecified or other languages
        if "```" in response_str:
             return response_str.split("```")[1].strip()
        return response_str.strip()

OLD_PY_SDK_KEYWORDS = {
    'google.generativeai',
    'GenerativeModel',
    'GenerationConfig',
    'model.start_chat',
    'model.generate_content',
}

def check_sdk_version_py(code_str):
    if any(keyword in code_str for keyword in OLD_PY_SDK_KEYWORDS):
        return 'old_sdk'
    if 'from google import genai' in code_str or 'import google.genai' in code_str:
        return 'new_sdk'
    return 'no_sdk'

OLD_TS_SDK_KEYWORDS = {
    '@google/generative-ai',
    'GoogleGenerativeAI',
    'getGenerativeModel',
    'generationConfig',
    'model.startChat',
    'model.generateContent',
}

def check_sdk_version_ts(code_str):
    if any(keyword in code_str for keyword in OLD_TS_SDK_KEYWORDS):
        return 'old_sdk'
    if '@google/genai' in code_str:
        return 'new_sdk'
    return 'no_sdk'

def check_model_version(code_str: str) -> str:
    """Check if code uses deprecated or current models."""
    code_lower = code_str.lower()
    
    for model in DEPRECATED_MODELS:
        if model.lower() in code_lower:
            return 'deprecated_model'
    
    for model in CURRENT_MODELS:
        if model.lower() in code_lower:
            return 'current_model'
    
    return 'no_model'

def analyze_code(code: str, lang: str) -> dict:
    """Analyze code for SDK and model version."""
    if lang == 'python':
        sdk_version = check_sdk_version_py(code)
    elif lang == 'typescript':
        sdk_version = check_sdk_version_ts(code)
    else:
        sdk_version = 'unknown_lang'
    
    model_version = check_model_version(code)
    
    return {
        'sdk_version': sdk_version,
        'model_version': model_version,
        'sdk_passed': sdk_version == 'new_sdk',
        'model_passed': model_version != 'deprecated_model'
    }

# --- Harness Core ---

def setup_directories():
    if os.path.exists(GENERATED_DIR):
        shutil.rmtree(GENERATED_DIR)
    os.makedirs(GENERATED_DIR)

def load_prompts() -> List[Dict[str, Any]]:
    with open(PROMPTS_FILE, 'r') as f:
        return json.load(f)

async def generate_code(prompt:str, language: str, client:genai.Client, mcp_session:Optional[ClientSession], retries=3, model_name=DEFAULT_MODEL) -> str:
    print(f"Generating code for ({language}): {prompt[:50]}...")

    tools = [mcp_session] if mcp_session else None

    for attempt in range(retries + 1):
        try:
            # Use a model that's good at coding. 
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                  tools=tools,
                  http_options={'timeout': 120000} # 2 minutes timeout
                )
            )
            return extract_code(response.text, language)
        except Exception as e:
            error_str = str(e)
            # Retry on 5XX errors or generic "Internal" errors
            is_5xx = any(code in error_str for code in ["500", "502", "503", "504", "Internal", "DeadlineExceeded"])
            if attempt < retries and is_5xx:
                wait_time = (attempt + 1) * 2 # Simple backoff: 2s, 4s, 6s
                print(f"  WARNING: API error (attempt {attempt+1}/{retries+1}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                if attempt == retries and is_5xx:
                     print(f"  ERROR: Failed after {retries+1} attempts.")
                raise e
    return ""

def save_code(code: str, test_id: str, language: str, output_dir: str) -> str:
    ext = "py" if language == "python" else "ts"
    file_path = os.path.join(output_dir, f"{test_id}.{ext}")
    with open(file_path, 'w') as f:
        f.write(code)
    return file_path

def execute_code(file_path: str) -> tuple[str, str, int]:
    print(f"Executing: {file_path}...")
    # Crucial: Add current directory to PYTHONPATH so 'python3 -m gemini_docs_mcp.server' works
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    
    try:
        # 30 second timeout to prevent hangs
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=60
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Execution timed out (60s)", -1
    except Exception as e:
        return "", str(e), -1

def validate_execution_result(stdout: str, stderr: str, returncode: int) -> bool:
    if returncode != 0:
        print(f"  FAILED (Execution): Non-zero return code {returncode}")
        print(f"  STDERR: {stderr.strip()}")
        return False
    return True
def get_available_skills() -> Dict[str, str]:
    """Get list of available skills with their descriptions."""
    skills = {}
    skill_dir = SKILL_PATH
    skill_file = os.path.join(skill_dir, "SKILL.md")
    
    if os.path.exists(skill_file):
        with open(skill_file, 'r') as f:
            content = f.read()
            # Extract name and description from YAML frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    for line in frontmatter.strip().split('\n'):
                        if line.startswith('name:'):
                            name = line.split(':', 1)[1].strip().strip('"\'')
                        if line.startswith('description:'):
                            desc = line.split(':', 1)[1].strip().strip('"\'')
                    skills[name] = desc
    return skills


def load_skill(name: str) -> str:
    """Load skill content by name. Returns the full SKILL.md content."""
    skill_file = os.path.join(SKILL_PATH, "SKILL.md")
    
    if os.path.exists(skill_file):
        with open(skill_file, 'r') as f:
            content = f.read()
            # Remove YAML frontmatter for cleaner output
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    return parts[2].strip()
            return content
    return f"Skill '{name}' not found."


def build_skill_system_instruction() -> str:
    """Build the system instruction with available skills."""
    skills = get_available_skills()
    skill_list = "\n".join([f"  <skill>\n    <name>{name}</name>\n    <description>{desc}</description>\n    <location>{SKILL_PATH}/SKILL.md</location>\n  </skill>" for name, desc in skills.items()])
    
    return f"""
# Available Agent Skills

You have access to the following specialized skills. To activate a skill and receive its detailed instructions, call the `activate_skill` tool with the skill's name.

<available_skills>
{skill_list}
</available_skills>""".strip()


# Define the activate_skill function for Gemini function calling
activate_skill_function = genai.types.FunctionDeclaration(
    name="activate_skill",
    description="Load a skill's instructional content into context. Available skills are listed in the system message. Pass the skill name to load its content.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the skill to load"
            }
        },
        "required": ["name"]
    }
)

# Define fetch_url function for fetching web content (e.g., llms.txt)
fetch_url_function = genai.types.FunctionDeclaration(
    name="fetch_url",
    description="Fetch content from a URL.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch content from"
            }
        },
        "required": ["url"]
    }
)


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch content from a URL with timeout."""
    import urllib.request
    import urllib.error
    import socket
    
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content = response.read().decode('utf-8')
            # Truncate very long responses
            if len(content) > 50000:
                return content[:50000] + f"\n\n[Truncated - original length: {len(content)}]"
            return content
    except socket.timeout:
        return f"Error: Request timed out after {timeout}s"
    except urllib.error.URLError as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error: {e}"


async def generate_code_with_skill(prompt: str, language: str, client: genai.Client, model: str, output_dir: str, max_steps: int = 8) -> str:
    """Generate code using Gemini API with skill function calling."""
    print(f"Generating code via API (model: {model}) for ({language}): {prompt[:50]}...")
    
    system_instruction = build_skill_system_instruction()
    tools = [genai.types.Tool(function_declarations=[activate_skill_function, fetch_url_function])]
    
    config = genai.types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        http_options={'timeout': 120000} # 2 minutes timeout
    )
    
    # Initial request
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    
    # Handle function calls with max steps limit
    # Build message history using actual Content objects to preserve thought_signature
    messages = [genai.types.Content(role="user", parts=[genai.types.Part(text=prompt)])]
    step = 0
    
    while step < max_steps and response.candidates and response.candidates[0].content.parts:
        # Check if any part is a function call
        has_function_call = False
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                break
        
        if not has_function_call:
            # It's a text response, we're done
            break
        
        step += 1
        print(f"  [Step {step}/{max_steps}]")
        
        # Add the model's full response (preserves thought_signature)
        messages.append(response.candidates[0].content)
        
        # Process all function calls in this response
        function_response_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                func_call = part.function_call
                
                # Dispatch to the correct function
                if func_call.name == "activate_skill":
                    arg_value = func_call.args.get('name', 'unknown')
                    print(f"    -> activate_skill: {arg_value}")
                    result = load_skill(arg_value)
                elif func_call.name == "fetch_url":
                    arg_value = func_call.args.get('url', '')
                    print(f"    -> fetch_url: {arg_value[:60]}...")
                    result = fetch_url(arg_value)
                else:
                    print(f"    -> unknown function: {func_call.name}")
                    result = f"Unknown function: {func_call.name}"
                
                # Create function response part
                function_response_parts.append(
                    genai.types.Part.from_function_response(
                        name=func_call.name,
                        response={"content": result}
                    )
                )
        
        # Add function responses as user message
        messages.append(genai.types.Content(role="user", parts=function_response_parts))
        
        # Continue the conversation
        response = await client.aio.models.generate_content(
            model=model,
            contents=messages,
            config=config
        )
    
    if step >= max_steps:
        print(f"  WARNING: Reached max steps ({max_steps}), stopping function call loop")
    
    # Log conversation history to file for debugging
    log_conversation_history(messages, response, prompt, output_dir)
    
    return extract_code(response.text, language)


def log_conversation_history(messages: list, response, prompt: str, output_dir: str) -> None:
    """Log conversation history to file for debugging."""
    log_file = os.path.join(output_dir, "skill_conversation_log.jsonl")
    
    # Track which functions were called and build full turn history
    functions_called = []
    turns = []
    
    for msg in messages:
        turn = {"role": msg.role, "parts": []}
        for part in msg.parts:
            if hasattr(part, 'text') and part.text:
                turn["parts"].append({"type": "text", "content": part.text[:500] + "..." if len(part.text) > 500 else part.text})
            elif hasattr(part, 'function_call') and part.function_call:
                functions_called.append(part.function_call.name)
                turn["parts"].append({
                    "type": "function_call",
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args) if part.function_call.args else {}
                })
            elif hasattr(part, 'function_response') and part.function_response:
                # Truncate long function responses
                resp_content = str(part.function_response.response)
                turn["parts"].append({
                    "type": "function_response", 
                    "name": part.function_response.name,
                    "response_length": len(resp_content)
                })
        turns.append(turn)
    
    # Build log entry with both summary and full history
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "skills_called": "activate_skill" in functions_called,
        "fetch_called": "fetch_url" in functions_called,
        "tools_called": functions_called,
        "turns": turns,
        "response_length": len(response.text) if response.text else 0
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


async def main():
    start_time = datetime.datetime.now()
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    
    parser = argparse.ArgumentParser(description="Gemini Docs MCP Eval Harness")
    parser.add_argument('--mode', choices=['execute', 'static', 'skill', 'vanilla'], default='static', 
                        help='Evaluation mode: static (MCP+SDK check), execute (MCP+run code), skill (API+function calling), vanilla (No MCP/Tools)')
    parser.add_argument('--max', type=int, help='Maximum number of tests to run')
    parser.add_argument('--ids', nargs='+', help='Specific test IDs to run')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, 
                        help=f'Model for evaluation (default: {DEFAULT_MODEL})')
    parser.add_argument('--output-dir', type=str, help='Directory to save results. If provided, enables resumable runs.')
    args = parser.parse_args()

    # Determine run directory
    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = os.path.join("tests", "runs", f"{timestamp_str}_{args.mode}_{args.model}")
    
    generated_dir = os.path.join(run_dir, "generated")
    result_file = os.path.join(run_dir, "result.json")
    
    os.makedirs(generated_dir, exist_ok=True)
    print(f"=== Starting Evaluation (Mode: {args.mode}) ===")
    print(f"Run Output Directory: {run_dir}")

    prompts = load_prompts()
    
    # Filter prompts
    if args.ids:
        prompts = [p for p in prompts if p.get('id') in args.ids]
    if args.max:
        prompts = prompts[:args.max]
    
    client = genai.Client()
    
    results = {}
    
    print(f"=== Starting Evaluation (Mode: {args.mode}) ===")

    # Skill mode: use Gemini API with skill function calling
    if args.mode == 'skill':
        print(f"Using skill from: {SKILL_PATH}")
        print(f"Model: {args.model}")
        
        for test_case in prompts:
            test_id = test_case.get('id', 'unknown')
            language = test_case.get('language', 'python')
            
            # Check if code file already exists as a proxy for "done"
            ext = "py" if language == "python" else "ts"
            expected_script = os.path.join(generated_dir, f"{test_id}.{ext}")
            
            if os.path.exists(expected_script) and args.output_dir:
                 print(f"Skipping {test_id} (already exists)")
                 with open(expected_script, 'r') as f:
                     code = f.read()
                 analysis = analyze_code(code, language)
                 passed = analysis['sdk_passed']
                 results[test_id] = {
                    "passed": passed, 
                    "analysis": analysis, 
                    "script": expected_script
                 }
                 continue
            
            print(f"\nTest: {test_id} ({language})")
            try:
                code = await generate_code_with_skill(test_case['prompt'], language, client, args.model, run_dir)
                script_path = save_code(code, test_id, language, generated_dir)
                
                analysis = analyze_code(code, language)
                passed = analysis['sdk_passed']
                results[test_id] = {
                    "passed": passed, 
                    "analysis": analysis, 
                    "script": script_path
                }
                status = f"sdk:{analysis['sdk_version']}, model:{analysis['model_version']}"
                print(f"  Analysis: {status} -> {'PASSED' if passed else 'FAILED'}")

            except Exception as e:
                print(f"  ERROR during test execution: {e}")
                results[test_id] = {"passed": False, "error": str(e)}

    # Vanilla mode: use Gemini API without MCP or tools
    elif args.mode == 'vanilla':
        print(f"Using model: {args.model} (No Tools/MCP)")
        
        for test_case in prompts:
            test_id = test_case.get('id', 'unknown')
            language = test_case.get('language', 'python')
            
            # Check if code file already exists as a proxy for "done"
            ext = "py" if language == "python" else "ts"
            expected_script = os.path.join(generated_dir, f"{test_id}.{ext}")
            
            if os.path.exists(expected_script) and args.output_dir:
                 print(f"Skipping {test_id} (already exists)")
                 with open(expected_script, 'r') as f:
                     code = f.read()
                 analysis = analyze_code(code, language)
                 passed = analysis['sdk_passed']
                 results[test_id] = {
                    "passed": passed, 
                    "analysis": analysis, 
                    "script": expected_script
                 }
                 continue
            
            print(f"\nTest: {test_id} ({language})")
            try:
                code = await generate_code(test_case['prompt'], language, client, None, model_name=args.model)
                script_path = save_code(code, test_id, language, generated_dir)
                
                analysis = analyze_code(code, language)
                passed = analysis['sdk_passed']
                results[test_id] = {
                    "passed": passed, 
                    "analysis": analysis, 
                    "script": script_path
                }
                status = f"sdk:{analysis['sdk_version']}, model:{analysis['model_version']}"
                print(f"  Analysis: {status} -> {'PASSED' if passed else 'FAILED'}")

            except Exception as e:
                print(f"  ERROR during test execution: {e}")
                results[test_id] = {"passed": False, "error": str(e)}
    
    # MCP modes: use MCP server
    else:
        async with stdio_client(server_params) as (read, write):
          async with ClientSession(read, write) as session:
            await session.initialize()
            for test_case in prompts:
                test_id = test_case.get('id', 'unknown')
                language = test_case.get('language', 'python')
                
                print(f"\nTest: {test_id} ({language})")
                try:
                    code = await generate_code(test_case['prompt'], language, client, session, model_name=args.model)
                    script_path = save_code(code, test_id, language, generated_dir)
                    
                    if args.mode == 'static':
                        analysis = analyze_code(code, language)
                        passed = analysis['sdk_passed']
                        results[test_id] = {"passed": passed, "analysis": analysis, "script": script_path}
                        status = f"sdk:{analysis['sdk_version']}, model:{analysis['model_version']}"
                        print(f"  Analysis: {status} -> {'PASSED' if passed else 'FAILED'}")

                    elif args.mode == 'execute':
                        if language == 'python':
                            stdout, stderr, returncode = execute_code(script_path)
                            passed = validate_execution_result(stdout, stderr, returncode)
                            results[test_id] = {"passed": passed, "script": script_path}
                            print(f"  Execution -> {'PASSED' if passed else 'FAILED'}")
                        else:
                            print(f"  SKIPPED (Execution not supported for {language})")
                            results[test_id] = {"passed": None, "status": "skipped_execution"}

                except Exception as e:
                    print(f"  ERROR during test execution: {e}")
                    results[test_id] = {"passed": False, "error": str(e)}

    print("\n=== Evaluation Summary ===")
    passed_count = sum(1 for r in results.values() if r.get('passed') is True)
    failed_count = sum(1 for r in results.values() if r.get('passed') is False)
    skipped_count = sum(1 for r in results.values() if r.get('passed') is None)
    total_count = len(results)
    
    print(f"Total Tests: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")

    # Calculate category stats
    category_stats = {}
    for test_id, result in results.items():
        test_case = next((p for p in prompts if p.get('id') == test_id), {})
        category = test_case.get('type', 'unknown')
        
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        
        stats = category_stats[category]
        stats["total"] += 1
        
        if result.get('passed') is True:
            stats["passed"] += 1
        elif result.get('passed') is False:
            stats["failed"] += 1
        else:
            stats["skipped"] += 1

    # Generate result.json
    failures = []
    for test_id, result in results.items():
        if result.get('passed') is False:
            # Find the prompt details
            test_case = next((p for p in prompts if p.get('id') == test_id), {})
            failure_entry = {
                "id": test_id,
                "language": test_case.get('language', 'unknown'),
                "type": test_case.get('type', 'unknown'),
                "prompt": test_case.get('prompt', 'unknown'),
                "error": result.get('error'),
                "analysis": result.get('analysis'),
                "script": result.get('script')
            }
            failures.append(failure_entry)

    report = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": args.model,
            "mode": args.mode,
            "duration_seconds": (datetime.datetime.now() - start_time).total_seconds()
        },
        "summary": {
            "total": total_count,
            "passed": passed_count,
            "failed": failed_count,
            "skipped": skipped_count
        },
        "by_category": category_stats,
        "failures": failures
    }

    with open(result_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # Exit with error if any failed (ignoring skipped)
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
