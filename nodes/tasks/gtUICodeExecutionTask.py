import ast
import json
from typing import Any, Tuple

from griptape.artifacts import BlobArtifact, TextArtifact, ErrorArtifact
from griptape.tasks import CodeExecutionTask

from ...py.griptape_settings import GriptapeSettings
from ..agent.gtComfyAgent import gtComfyAgent as Agent
from .gtUIBaseTask import gtUIBaseTask


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class DangerousCodeChecker(ast.NodeVisitor):
    allowed_imports = {
        # Date and Time
        "datetime",
        "time",
        "calendar",
        "zoneinfo",
        # Text Processing
        "re",
        "string",
        "difflib",
        "textwrap",
        "unicodedata",
        # Data Formats
        "json",
        "csv",
        "yaml",  # if pyyaml is installed
        "toml",  # if toml is installed
        "xml.etree.ElementTree",
        "xml.dom.minidom",
        "html.parser",
        # Math and Numbers
        "math",
        "decimal",
        "fractions",
        "numbers",
        "statistics",
        "random",
        # Data Structures and Algorithms
        "collections",
        "heapq",
        "bisect",
        "array",
        "queue",
        "dataclasses",
        # Functional Programming
        "itertools",
        "functools",
        "operator",
        # File and Path Operations (safe subset)
        "pathlib",
        "os.path",
        "fileinput",
        "configparser",
        "zipfile",
        "tarfile",
        "gzip",
        "bz2",
        "lzma",
        # Input/Output
        "io.StringIO",
        "io.BytesIO",
        "io.TextIOWrapper",
        # Data Types
        "typing",
        "enum",
        "uuid",
        # Encoding/Decoding
        "base64",
        "binascii",
        "codecs",
        "urllib.parse",
        # Development Tools
        "pdb",
        "unittest",
        "doctest",
        "timeit",
        # Formatting and Printing
        "pprint",
        "reprlib",
        "rich",
        # Commonly Used Third-Party Libraries (if installed)
        "numpy",  # Numerical computing
        "pandas",  # Data analysis
        "scipy",  # Scientific computing
        "matplotlib",  # Plotting
        "seaborn",  # Statistical visualization
        "PIL.Image",  # Image processing
        "PIL.ImageDraw",
        "PIL.ImageFont",
        # Compression and Archiving
        "zlib",
        "hashlib",  # Cryptographic hashing
        # Templating
        "string.Template",
        "jinja2",  # if installed
        # Logging
        "logging",
        "logging.config",
        "logging.handlers",
    }

    def __init__(self):
        super().__init__()
        self.unapproved_imports = []

        # Create a mapping of module to allowed submodules
        self.allowed_submodules = {}
        for imp in self.allowed_imports:
            parts = imp.split(".")
            if len(parts) > 1:
                if parts[0] not in self.allowed_submodules:
                    self.allowed_submodules[parts[0]] = set()
                self.allowed_submodules[parts[0]].add(parts[1])

    def visit_Import(self, node):
        for alias in node.names:
            # Check if the full import path is allowed
            if alias.name not in self.allowed_imports:
                # Check if it's a submodule of an allowed import
                parts = alias.name.split(".")
                if not (
                    len(parts) > 1
                    and parts[0] in self.allowed_submodules
                    and parts[1] in self.allowed_submodules[parts[0]]
                ):
                    self.unapproved_imports.append(alias.name)

    def visit_ImportFrom(self, node):
        if node.module is None:  # Handle relative imports
            self.unapproved_imports.append("relative import")
            return

        # If the module itself is in allowed_imports, everything is fine
        if node.module in self.allowed_imports:
            return

        # Check if we're importing from a module that has allowed submodules
        if node.module in self.allowed_submodules:
            for alias in node.names:
                # Check if the specific import is allowed
                if alias.name not in self.allowed_submodules[node.module]:
                    self.unapproved_imports.append(f"{node.module}.{alias.name}")
        else:
            # The module isn't in our allowed list at all
            self.unapproved_imports.append(node.module)

    def visit_Call(self, node):
        # Check for dangerous calls
        if isinstance(node.func, ast.Name) and node.func.id in {"exec", "__import__"}:
            self.unapproved_imports.append(f"{node.func.id}() call")
        self.generic_visit(node)


def check_script_for_danger(script):
    try:
        # Parse the script into an AST
        tree = ast.parse(script)
        checker = DangerousCodeChecker()
        checker.visit(tree)

        # If no unapproved imports are found, the script is safe
        is_safe = len(checker.unapproved_imports) == 0
        return is_safe, checker.unapproved_imports
    except SyntaxError as e:
        # Return as unsafe if there is a syntax error
        return False, [f"Syntax error: {e}"]


any = AnyType("*")
examples = {
    "Word count": "output = len(input.split())",
    "Reverse the input": "output = input[::-1]",
    "Sort a list of numbers": "output = sorted([int(x) for x in input.split(',')])",
    "Sort a list of words": "output = sorted(input.split())",
    "Find unique words": "output = list(set(input.split()))",
    "Extract numbers": "output = re.findall(r'\\d+', input)",
    "Replace words": "output = input.replace('old', 'new')",
    "Find email addresses": "output = re.findall(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', input)",
    "Remove spaces and punctuation": """def clean_text(text):
    import string
    text = text.strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

output = clean_text(input)
""",
}

examples_keys_list = list(examples.keys())


def build_CodeExecutionTask(code: str, num_outputs: int = 3) -> CodeExecutionTask:
    function_name = "dynamic_task"
    exec_globals = {"TextArtifact": TextArtifact, "BlobArtifact": BlobArtifact, "json": json}
    exec_locals = {}
    indented_code = "\n".join([f"    {line}" for line in code.splitlines()])
    
    wrapped_code = f"""
def {function_name}(task):
    input_data = json.loads(task.input.value)
    input_0 = input_data["input_0"]
    input_1 = input_data["input_1"]
    input_2 = input_data["input_2"]
{indented_code}
    output_artifacts = []
    for i in range({num_outputs}):
        artifact = locals().get(f'output_{{i}}', '')
        if isinstance(artifact, str):
            output_artifacts.append(TextArtifact(artifact))
        elif isinstance(artifact, bytes):
            output_artifacts.append(BlobArtifact(artifact))
        else:
            output_artifacts.append(TextArtifact(str(artifact)))
    return output_artifacts
"""
    exec(wrapped_code, exec_globals, exec_locals)
    return CodeExecutionTask(on_run=exec_locals[function_name])



class gtUICodeExecutionTask(gtUIBaseTask):
    DESCRIPTION = "Executes python code as a task.\nThe code takes the `input` from the task and should define an `output` variable that will be returned as the task's output."
    CATEGORY = "Griptape/Code"
    OUTPUTS = ("STRING", "STRING", "STRING", "AGENT", "TASK")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update(
            {
                "input_0": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "Input text #0 (required).",
                        "default": "",
                    },
                ),
            }
        )
        del inputs["required"]["STRING"]
        del inputs["optional"]["input_string"]
        del inputs["optional"]["key_value_replacement"]
        
        inputs["optional"].update(
            {
                "input_1": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "Input text #1 (optional).",
                        "default": "",
                    },
                ),
                "input_2": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "Input text #2 (optional).",
                        "default": "",
                    },
                ),
                "examples": (
                    (),
                    {
                        "default": "Custom code",
                        "tooltip": "Select an example code snippet to replace the default code.",
                    },
                ),
                "code": (
                    "STRING",
                    {
                        "placeholder": "Python code to execute. \n`input` is any input text.\nDefine `output_0`, `output_1`, etc. for multiple outputs.\n\noutput_0 = input.upper()",
                        "default": """# Python code to execute.
# `input` is any input text.
# Define `output_0`, `output_1`, output_2. for multiple outputs

output_0 = input_0.upper()
output_1 = input_1.lower()
output_2 = input_2[::-1]
""",
                        "multiline": True,
                        "tooltip": """Python code to execute. 
The code can define multiple inputs and output variables (output_0, output_1, etc.) that will be returned as the task's outputs.

Example:
# Sorts and counts a list
output_0 = sorted(input.split())
output_1 = len(output_0)
""",
                    },
                ),
            }
        )
        inputs["hidden"] = {"unique_id": "UNIQUE_ID"}
        return inputs
        
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("Output_0", "Output_1", "Output_2")
    FUNCTION = "run"
    CATEGORY = "Griptape/Code"

    def run(self, **kwargs) -> Tuple[Any, ...]:
        input_0 = kwargs.get("input_0", "")
        input_1 = kwargs.get("input_1", "")
        input_2 = kwargs.get("input_2", "")
        code = kwargs.get("code", "")
        agent = kwargs.get("agent", None)
        settings = GriptapeSettings()
        code_execution = settings.get_settings_key("allow_code_execution")
        code_execution_dangerous = settings.get_settings_key("allow_code_execution_dangerous")

        if not code_execution:
            return ("❌ Code execution is disabled.", "", "", None, None)

        if not agent:
            agent = Agent()

        if not code_execution_dangerous:
            safe_code, response = check_script_for_danger(code)
            if not safe_code:
                return (f"❌ Dangerous code detected: {response}", "", "", None, None)

        dynamic_task = build_CodeExecutionTask(code, num_outputs=3)

        # Serialize inputs to JSON string
        input_dict = {"input_0": input_0, "input_1": input_1, "input_2": input_2}

        try:
            agent.add_task(dynamic_task)
            input_str = json.dumps(input_dict)
            result = agent.run(input_str)
            outputs = result.output_task.output
            
            if isinstance(outputs, ErrorArtifact):
                return (f"❌ Task execution failed: {outputs.value}", "", "", None, None)


            output_values = [artifact.value if artifact else "" for artifact in outputs]
            while len(output_values) < 3:
                output_values.append("")  # pad if needed

            return (*output_values[:3], agent, None)

        except Exception as e:
            print(e)
            return (f"❌ Error: {str(e)}", "", "", None, None)

