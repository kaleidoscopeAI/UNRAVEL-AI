#!/usr/bin/env python3
"""
Kaleidoscope AI - Software Ingestion & Mimicry System
====================================================
A cutting-edge system that can ingest, analyze, and mimic any software by:
1. Decompiling binaries and obfuscated code
2. Creating specifications from analyzed code
3. Reconstructing software with enhanced capabilities
4. Generating new software based on learned patterns

This pushes the boundaries of software analysis and generation through
the clever application of graph theory, machine learning, and automated 
binary analysis.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import json
import re
import logging
import argparse
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FileType(Enum):
    """Enum representing different file types for processing"""
    BINARY = "binary"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    JAVA = "java"
    ASSEMBLY = "assembly"
    UNKNOWN = "unknown"

class KaleidoscopeCore:
    """Core system coordinator for Kaleidoscope AI"""
    
    def __init__(self):
        self.modules = {}
        self.state = {}
        self._initialize_logging()
        
    def _initialize_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("kaleidoscope.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def register_module(self, name: str, module: Any):
        """Register a module with the core system"""
        if name in self.modules:
            logger.warning(f"Replacing existing module {name}")
        self.modules[name] = module
        logger.info(f"Registered module: {name}")

    def __init__(self, 
                 work_dir: str = None, 
                 llm_endpoint: str = "http://localhost:8000/v1",
                 max_workers: int = 4):
        """
        Initialize the Kaleidoscope core system
        
        Args:
            work_dir: Working directory for processing files
            llm_endpoint: Endpoint for LLM API access
            max_workers: Maximum number of concurrent workers
        """
        self.work_dir = work_dir or os.path.join(os.getcwd(), "kaleidoscope_workdir")
        self.source_dir = os.path.join(self.work_dir, "source")
        self.decompiled_dir = os.path.join(self.work_dir, "decompiled")
        self.specs_dir = os.path.join(self.work_dir, "specs")
        self.reconstructed_dir = os.path.join(self.work_dir, "reconstructed")
        self.llm_endpoint = llm_endpoint
        self.max_workers = max_workers
        self.dependency_graph = nx.DiGraph()
        
        # Create necessary directories
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.decompiled_dir, exist_ok=True)
        os.makedirs(self.specs_dir, exist_ok=True)
        os.makedirs(self.reconstructed_dir, exist_ok=True)
        
        # Check for required tools
        self._check_required_tools()
    
    def _check_required_tools(self) -> None:
        """Check if required external tools are available"""
        tools_to_check = {
            "radare2": "r2",
            "ghidra_server": "ghidra_server",
            "retdec-decompiler": "retdec-decompiler",
            "js-beautify": "js-beautify",
        }
        
        missing_tools = []
        for tool_name, cmd in tools_to_check.items():
            if not shutil.which(cmd):
                missing_tools.append(tool_name)
        
        if missing_tools:
            logger.warning(f"Missing tools: {', '.join(missing_tools)}")
            logger.info("Some functionality may be limited. Install missing tools for full capabilities.")
            
    def detect_file_type(self, file_path: str) -> FileType:
        """
        Detect the type of the input file
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            FileType: The detected file type
        """
        # Check file extension first
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Map extensions to file types
        ext_map = {
            ".exe": FileType.BINARY,
            ".dll": FileType.BINARY,
            ".so": FileType.BINARY,
            ".dylib": FileType.BINARY,
            ".js": FileType.JAVASCRIPT,
            ".mjs": FileType.JAVASCRIPT,
            ".py": FileType.PYTHON,
            ".cpp": FileType.CPP,
            ".cc": FileType.CPP,
            ".c": FileType.C,
            ".cs": FileType.CSHARP,
            ".java": FileType.JAVA,
            ".asm": FileType.ASSEMBLY,
            ".s": FileType.ASSEMBLY
        }
        
        if file_ext in ext_map:
            return ext_map[file_ext]
        
        # If extension doesn't match, try to detect file type using magic/file command
        try:
            file_output = subprocess.check_output(["file", file_path], universal_newlines=True)
            
            if "ELF" in file_output or "PE32" in file_output or "Mach-O" in file_output:
                return FileType.BINARY
            elif "JavaScript" in file_output:
                return FileType.JAVASCRIPT
            elif "Python" in file_output:
                return FileType.PYTHON
            elif "C++ source" in file_output:
                return FileType.CPP
            elif "C source" in file_output:
                return FileType.C
            elif "assembler source" in file_output:
                return FileType.ASSEMBLY
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Could not use 'file' command to detect file type")
        
        return FileType.UNKNOWN

    def ingest_software(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point for software ingestion
        
        Args:
            file_path: Path to the software file to ingest
            
        Returns:
            Dict: Results of ingestion process
        """
        logger.info(f"Starting ingestion of {file_path}")
        
        # Copy source file to work directory
        source_filename = os.path.basename(file_path)
        source_dest = os.path.join(self.source_dir, source_filename)
        shutil.copy2(file_path, source_dest)
        
        # Detect file type
        file_type = self.detect_file_type(source_dest)
        logger.info(f"Detected file type: {file_type.value}")
        
        # Set up result dictionary
        result = {
            "original_file": file_path,
            "work_file": source_dest,
            "file_type": file_type.value,
            "decompiled_files": [],
            "spec_files": [],
            "reconstructed_files": [],
            "status": "pending"
        }
        
        # Process based on file type
        try:
            if file_type == FileType.BINARY:
                result = self._process_binary(source_dest, result)
            elif file_type == FileType.JAVASCRIPT:
                result = self._process_javascript(source_dest, result)
            elif file_type == FileType.PYTHON:
                result = self._process_python(source_dest, result)
            elif file_type in [FileType.C, FileType.CPP]:
                result = self._process_c_cpp(source_dest, result)
            else:
                logger.warning(f"Unsupported file type: {file_type.value}")
                result["status"] = "unsupported_file_type"
                return result
            
            # Generate specifications
            if result["decompiled_files"]:
                spec_files = self._generate_specifications(result["decompiled_files"])
                result["spec_files"] = spec_files
                
                # Reconstruct software
                if spec_files:
                    reconstructed_files = self._reconstruct_software(spec_files)
                    result["reconstructed_files"] = reconstructed_files
                    result["status"] = "completed"
                else:
                    result["status"] = "failed_spec_generation"
            else:
                result["status"] = "failed_decompilation"
                
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)
            
        return result

    def _process_binary(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a binary file using radare2 and ghidra"""
        logger.info(f"Processing binary: {file_path}")
        
        decompiled_dir = os.path.join(self.decompiled_dir, os.path.basename(file_path))
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Use radare2 for initial analysis
        radare_output = os.path.join(decompiled_dir, "radare_analysis.txt")
        try:
            with open(radare_output, 'w') as f:
                # Basic analysis with radare2
                subprocess.run(
                    ["r2", "-q", "-c", "aaa; s main; pdf", file_path],
                    stdout=f, 
                    stderr=subprocess.PIPE,
                    check=True
                )
            result["decompiled_files"].append(radare_output)
            logger.info(f"Radare2 analysis complete: {radare_output}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Radare2 analysis failed: {str(e)}")
        
        # Use RetDec for C decompilation
        retdec_output = os.path.join(decompiled_dir, "retdec_decompiled.c")
        try:
            subprocess.run(
                ["retdec-decompiler", file_path, "-o", retdec_output],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            result["decompiled_files"].append(retdec_output)
            logger.info(f"RetDec decompilation complete: {retdec_output}")
        except subprocess.CalledProcessError as e:
            logger.error(f"RetDec decompilation failed: {str(e)}")
            
        # If we have at least one successful decompilation, continue
        if not result["decompiled_files"]:
            logger.error(f"All decompilation methods failed for {file_path}")
            
        return result

    def _process_javascript(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JavaScript file"""
        logger.info(f"Processing JavaScript: {file_path}")
        
        decompiled_dir = os.path.join(self.decompiled_dir, os.path.basename(file_path))
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Beautify and analyze JavaScript
        beautified_output = os.path.join(decompiled_dir, "beautified.js")
        try:
            subprocess.run(
                ["js-beautify", file_path, "-o", beautified_output],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            result["decompiled_files"].append(beautified_output)
            logger.info(f"JavaScript beautification complete: {beautified_output}")
            
            # Extract structure using AST analysis
            ast_output = os.path.join(decompiled_dir, "ast_analysis.json")
            self._generate_js_ast(beautified_output, ast_output)
            if os.path.exists(ast_output):
                result["decompiled_files"].append(ast_output)
                logger.info(f"JavaScript AST analysis complete: {ast_output}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"JavaScript processing failed: {str(e)}")
            
        return result
    
    def _generate_js_ast(self, js_file: str, output_file: str) -> None:
        """Generate AST analysis for JavaScript file"""
        # Create a temporary Node.js script to extract AST
        with tempfile.NamedTemporaryFile(suffix='.js', mode='w', delete=False) as temp:
            temp.write("""
            const fs = require('fs');
            const acorn = require('acorn');
            
            const sourceCode = fs.readFileSync(process.argv[2], 'utf8');
            
            try {
                const ast = acorn.parse(sourceCode, {
                    ecmaVersion: 'latest',
                    sourceType: 'module',
                    locations: true
                });
                
                // Extract function names, structure, and calls
                const functions = [];
                const imports = [];
                const classes = [];
                
                function traverseNode(node, parent) {
                    if (node.type === 'FunctionDeclaration' || node.type === 'ArrowFunctionExpression') {
                        const name = node.id ? node.id.name : 'anonymous';
                        functions.push({
                            name: name,
                            params: node.params.map(p => p.type === 'Identifier' ? p.name : 'complexParam'),
                            loc: node.loc,
                            parent: parent
                        });
                    } else if (node.type === 'ImportDeclaration') {
                        imports.push({
                            source: node.source.value,
                            specifiers: node.specifiers.map(s => {
                                if (s.type === 'ImportDefaultSpecifier') {
                                    return { type: 'default', name: s.local.name };
                                } else if (s.type === 'ImportSpecifier') {
                                    return { type: 'named', name: s.local.name, imported: s.imported.name };
                                } else {
                                    return { type: 'namespace', name: s.local.name };
                                }
                            })
                        });
                    } else if (node.type === 'ClassDeclaration') {
                        classes.push({
                            name: node.id.name,
                            methods: node.body.body.filter(m => m.type === 'MethodDefinition').map(m => ({
                                name: m.key.name,
                                kind: m.kind,
                                static: m.static
                            })),
                            loc: node.loc
                        });
                    }
                    
                    // Recursively traverse child nodes
                    for (const key in node) {
                        if (node[key] && typeof node[key] === 'object') {
                            if (Array.isArray(node[key])) {
                                node[key].forEach(child => {
                                    if (child && typeof child === 'object') {
                                        traverseNode(child, node.type);
                                    }
                                });
                            } else if (node[key].type) {
                                traverseNode(node[key], node.type);
                            }
                        }
                    }
                }
                
                traverseNode(ast, null);
                
                const analysis = {
                    functions,
                    imports,
                    classes,
                    nodeCount: countNodes(ast)
                };
                
                function countNodes(node) {
                    let count = 1;
                    for (const key in node) {
                        if (node[key] && typeof node[key] === 'object') {
                            if (Array.isArray(node[key])) {
                                node[key].forEach(child => {
                                    if (child && typeof child === 'object') {
                                        count += countNodes(child);
                                    }
                                });
                            } else if (node[key].type) {
                                count += countNodes(node[key]);
                            }
                        }
                    }
                    return count;
                }
                
                fs.writeFileSync(process.argv[3], JSON.stringify(analysis, null, 2));
                console.log('AST analysis complete');
            } catch (error) {
                console.error('Error parsing JavaScript:', error.message);
                process.exit(1);
            }
            """)
            temp_path = temp.name
        
        try:
            # Check if acorn is installed
            subprocess.run(
                ["npm", "list", "-g", "acorn"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            # Run the AST extraction script
            subprocess.run(
                ["node", temp_path, js_file, output_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"JavaScript AST extraction failed: {str(e)}")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _process_python(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a Python file for analysis"""
        logger.info(f"Processing Python: {file_path}")
        
        decompiled_dir = os.path.join(self.decompiled_dir, os.path.basename(file_path))
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Generate AST analysis for Python
        ast_output = os.path.join(decompiled_dir, "python_ast_analysis.json")
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp:
                temp.write("""
import ast
import json
import sys

def analyze_python_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    try:
        tree = ast.parse(source_code)
        
        # Extract functions, classes, imports
        functions = []
        classes = []
        imports = []
        global_vars = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', None),
                })
            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'args': [arg.arg for arg in item.args.args],
                            'lineno': item.lineno,
                        })
                
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'bases': [base.id if isinstance(base, ast.Name) else 'complex_base' for base in node.bases],
                    'lineno': node.lineno,
                })
            elif isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        'name': name.name,
                        'asname': name.asname,
                        'lineno': node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                imports.append({
                    'module': node.module,
                    'names': [{'name': name.name, 'asname': name.asname} for name in node.names],
                    'lineno': node.lineno,
                })
            elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():  # Assume constants are UPPERCASE
                        global_vars.append({
                            'name': target.id,
                            'lineno': node.lineno,
                        })
        
        analysis = {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'global_vars': global_vars,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Python AST analysis complete")
        return True
    except SyntaxError as e:
        print(f"Error parsing Python file: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <python_file> <output_file>")
        sys.exit(1)
    
    success = analyze_python_file(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
                """)
                temp_path = temp.name
            
            # Run the Python AST analysis
            subprocess.run(
                [sys.executable, temp_path, file_path, ast_output],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            if os.path.exists(ast_output):
                result["decompiled_files"].append(ast_output)
                logger.info(f"Python AST analysis complete: {ast_output}")
                
                # Also add the original source since Python is already human-readable
                source_copy = os.path.join(decompiled_dir, "source.py")
                shutil.copy2(file_path, source_copy)
                result["decompiled_files"].append(source_copy)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Python analysis failed: {str(e)}")
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            
        return result

    def _process_c_cpp(self, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a C/C++ file"""
        logger.info(f"Processing C/C++: {file_path}")
        
        decompiled_dir = os.path.join(self.decompiled_dir, os.path.basename(file_path))
        os.makedirs(decompiled_dir, exist_ok=True)
        
        # Copy the source for reference
        source_copy = os.path.join(decompiled_dir, os.path.basename(file_path))
        shutil.copy2(file_path, source_copy)
        result["decompiled_files"].append(source_copy)
        
        # Try to parse and analyze C/C++ structure
        ast_output = os.path.join(decompiled_dir, "cpp_structure.json")
        
        try:
            # Use clang to get AST dump
            ast_dump = subprocess.check_output(
                ["clang", "-Xclang", "-ast-dump", "-fsyntax-only", file_path],
                universal_newlines=True,
                stderr=subprocess.PIPE
            )
            
            # Parse the AST dump to extract structure
            functions = []
            classes = []
            global_vars = []
            
            # Very simple regex-based extraction (a real implementation would use libclang)
            fn_pattern = r"FunctionDecl.*?([\w~]+)"
            class_pattern = r"CXXRecordDecl.*?class ([\w~]+)"
            var_pattern = r"VarDecl.*? ([\w~]+) '.*?'"
            
            for match in re.finditer(fn_pattern, ast_dump):
                functions.append({"name": match.group(1)})
            
            for match in re.finditer(class_pattern, ast_dump):
                classes.append({"name": match.group(1)})
                
            for match in re.finditer(var_pattern, ast_dump):
                global_vars.append({"name": match.group(1)})
            
            structure = {
                "functions": functions,
                "classes": classes,
                "global_vars": global_vars
            }
            
            with open(ast_output, 'w') as f:
                json.dump(structure, f, indent=2)
                
            result["decompiled_files"].append(ast_output)
            logger.info(f"C/C++ structure analysis complete: {ast_output}")
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"C/C++ analysis failed: {str(e)}")
            
        return result

    def _generate_specifications(self, decompiled_files: List[str]) -> List[str]:
        """
        Generate specifications from decompiled files using LLM
        
        Args:
            decompiled_files: List of paths to decompiled files
            
        Returns:
            List[str]: Paths to generated specification files
        """
        logger.info(f"Generating specifications from {len(decompiled_files)} decompiled files")
        
        # Create a unique specification folder
        timestamp = int(os.path.getmtime(decompiled_files[0]))
        spec_dir = os.path.join(self.specs_dir, f"spec_{timestamp}")
        os.makedirs(spec_dir, exist_ok=True)
        
        spec_files = []
        
        # Process with LLM to extract specifications
        combined_spec_path = os.path.join(spec_dir, "combined_spec.md")
        with open(combined_spec_path, 'w') as spec_file:
            spec_file.write("# Software Specification\n\n")
            spec_file.write("This document contains specifications extracted from the decompiled software.\n\n")
            
            # Process each decompiled file
            for decompiled_file in decompiled_files:
                file_name = os.path.basename(decompiled_file)
                spec_file.write(f"## {file_name}\n\n")
                
                with open(decompiled_file, 'r', errors='replace') as f:
                    try:
                        content = f.read()
                        
                        # Use Huntley's approach:
                        # 1. Split large files into chunks
                        # 2. Process each chunk with LLM
                        # 3. Combine into specifications
                        
                        # For now, simulate LLM processing
                        spec_file.write(f"### Features and Components\n\n")
                        
                        # Basic format detection
                        if decompiled_file.endswith('.json'):
                            try:
                                data = json.loads(content)
                                if "functions" in data:
                                    spec_file.write("#### Functions\n\n")
                                    for func in data["functions"]:
                                        name = func.get("name", "unknown")
                                        spec_file.write(f"- `{name}`\n")
                                
                                if "classes" in data:
                                    spec_file.write("\n#### Classes\n\n")
                                    for cls in data["classes"]:
                                        name = cls.get("name", "unknown")
                                        spec_file.write(f"- `{name}`\n")
                            except json.JSONDecodeError:
                                spec_file.write("Could not parse JSON structure.\n")
                        else:
                            # Very basic function extraction for demonstration
                            # In a real implementation, this would use the LLM
                            fn_matches = re.findall(r"(?:function|def|void|int|bool|string)\s+(\w+)\s*\(", content)
                            if fn_matches:
                                spec_file.write("#### Extracted Functions\n\n")
                                for fn in fn_matches[:20]:  # Limit to avoid overwhelming the spec
                                    spec_file.write(f"- `{fn}`\n")
                            
                        spec_file.write("\n")
                        
                    except Exception as e:
                        logger.error(f"Error processing {decompiled_file}: {str(e)}")
                        spec_file.write(f"Error processing file: {str(e)}\n\n")
        
        spec_files.append(combined_spec_path)
        logger.info(f"Generated specification: {combined_spec_path}")
        
        # Now generate Claude-style prompts (a la Huntley)
        with open(os.path.join(spec_dir, "prompt_template.txt"), 'w') as f:
            f.write("""
CLI.js is a commonjs typescript application which has been compiled with webpack.
The symbols have been stripped.
Inspect the source code thoroughly (extra thinking) but skip the SentrySDK source code.
Create a specification library of features of the application.
Convert the source code into human readable.
Keep going until you are done!

Now deobfuscate the application.
Split the application into separate files per domain in the SPECS folder.
Provide an overview of the directory structure before starting deobfuscation.
Skip the SENTRYSDK.
            """)
        
        # In a real implementation, these prompts would be used with the LLM
        
        return spec_files

    def _reconstruct_software(self, spec_files: List[str]) -> List[str]:
        """
        Reconstruct software from specifications
        
        Args:
            spec_files: List of paths to specification files
            
        Returns:
            List[str]: Paths to reconstructed software files
        """
        logger.info(f"Reconstructing software from {len(spec_files)} specification files")
        
        # Create a unique folder for reconstructed software
        timestamp = int(os.path.getmtime(spec_files[0]))
        reconstructed_dir = os.path.join(self.reconstructed_dir, f"reconstructed_{timestamp}")
        os.makedirs(reconstructed_dir, exist_ok=True)
        
        reconstructed_files = []
        
        # In a real implementation, this would use LLM to generate code from specs
        # For demonstration, we'll create a simulated project structure
        
        readme_path = os.path.join(reconstructed_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write("""# Reconstructed Software

This software has been automatically reconstructed by Kaleidoscope AI.

## Project Structure

```
src/
  ├── main.py           # Main entry point
  ├── core/             # Core functionality
  │   ├── __init__.py
  │   └── engine.py
  ├── utils/            # Utility functions
  │   ├── __init__.py
  │   └── helpers.py
  └── data/             # Data processing
      ├── __init__.py
      └── processor.py
```

## How to Run

```bash
python src/main.py
```
""")
        reconstructed_files.append(readme_path)