#!/usr/bin/env python3
"""
Kaleidoscope AI Web Interface
============================
A simple web interface for the Kaleidoscope AI system that allows users
to interact with the software ingestion and mimicry system through a browser.

This provides a more user-friendly interface compared to the command-line
chatbot and makes the powerful features more accessible.
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Web server imports
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Ensure the kaleidoscope_core module is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaleidoscope_core import KaleidoscopeCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_web.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), "uploads")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload size
app.config['SECRET_KEY'] = str(uuid.uuid4())

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state
kaleidoscope = None
task_queue = queue.Queue()
current_task = None
task_history = []
worker_thread = None
running = False

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    # Allow most executable and code file extensions
    allowed_extensions = {
        'exe', 'dll', 'so', 'dylib',  # Binaries
        'js', 'mjs',                  # JavaScript
        'py',                         # Python
        'c', 'cpp', 'h', 'hpp',       # C/C++
        'java', 'class', 'jar',       # Java
        'go',                         # Go
        'rs',                         # Rust
        'php',                        # PHP
        'rb',                         # Ruby
        'cs',                         # C#
        'asm', 's'                    # Assembly
    }
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def worker_loop():
    """Background worker thread to process tasks"""
    global current_task, running
    
    while running:
        try:
            # Get task from queue
            task = task_queue.get(timeout=1.0)
            
            # Set as current task
            current_task = task
            current_task["status"] = "processing"
            
            # Process task
            if task["type"] == "ingest":
                worker_ingest(task)
            elif task["type"] == "mimic":
                worker_mimic(task)
            
            # Mark task as done
            task_queue.task_done()
            
            # Add to history
            task_history.append(task)
            
            # Limit history size
            if len(task_history) > 10:
                task_history.pop(0)
                
            # Clear current task
            current_task = None
            
        except queue.Empty:
            # No tasks in queue
            pass
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            
            # Update current task status
            if current_task:
                current_task["status"] = "error"
                current_task["error"] = str(e)
                
                # Add to history
                task_history.append(current_task)
                
                # Clear current task
                current_task = None

def worker_ingest(task):
    """
    Worker function to ingest software
    
    Args:
        task: Task information
    """
    try:
        # Ingest software
        file_path = task["file_path"]
        
        logger.info(f"Ingesting {file_path}...")
        
        # Run ingestion
        result = kaleidoscope.ingest_software(file_path)
        
        # Update task with result
        task.update(result)
        task["status"] = result["status"]
        
        logger.info(f"Ingestion completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error in ingestion: {str(e)}")
        task["status"] = "error"
        task["error"] = str(e)

def worker_mimic(task):
    """
    Worker function to mimic software
    
    Args:
        task: Task information
    """
    try:
        # Mimic software
        spec_files = task["spec_files"]
        target_language = task["target_language"]
        
        logger.info(f"Generating mimicked version in {target_language}...")
        
        # Run mimicry
        result = kaleidoscope.mimic_software(spec_files, target_language)
        
        # Update task with result
        task.update(result)
        task["status"] = result["status"]
        
        logger.info(f"Mimicry completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error in mimicry: {str(e)}")
        task["status"] = "error"
        task["error"] = str(e)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for ingestion"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check if file is allowed
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Create task
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": "ingest",
        "file_path": file_path,
        "file_name": filename,
        "status": "queued",
        "timestamp": time.time()
    }
    
    # Add to queue
    task_queue.put(task)
    
    return jsonify({
        "status": "success",
        "message": f"File {filename} uploaded and queued for ingestion",
        "task_id": task_id
    })

@app.route('/mimic', methods=['POST'])
def mimic_software():
    """Handle request to mimic software"""
    data = request.json
    
    # Check required fields
    if not data or 'language' not in data or 'task_id' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Get source task
    source_task = None
    for task in task_history:
        if task.get("id") == data['task_id']:
            source_task = task
            break
    
    if not source_task or source_task["type"] != "ingest" or source_task["status"] != "completed":
        return jsonify({"error": "Source task not found or not completed"}), 400
    
    # Check if we have specification files
    if "spec_files" not in source_task or not source_task["spec_files"]:
        return jsonify({"error": "No specification files available"}), 400
    
    # Validate target language
    target_language = data['language'].lower()
    valid_languages = ["python", "javascript", "c", "cpp", "c++", "java"]
    
    if target_language not in valid_languages:
        return jsonify({"error": f"Unsupported language: {target_language}"}), 400
    
    # Map language aliases
    if target_language in ["c++", "cpp"]:
        target_language = "cpp"
    
    # Create task
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": "mimic",
        "source_task_id": data['task_id'],
        "spec_files": source_task["spec_files"],
        "target_language": target_language,
        "status": "queued",
        "timestamp": time.time()
    }
    
    # Add to queue
    task_queue.put(task)
    
    return jsonify({
        "status": "success",
        "message": f"Queued mimicry in {target_language}",
        "task_id": task_id
    })

@app.route('/status')
def get_status():
    """Get status of tasks"""
    # Prepare current task info if available
    current = None
    if current_task:
        current = {
            "id": current_task.get("id"),
            "type": current_task.get("type"),
            "status": current_task.get("status"),
            "file_name": current_task.get("file_name") if "file_name" in current_task else None,
            "target_language": current_task.get("target_language") if "target_language" in current_task else None,
            "timestamp": current_task.get("timestamp")
        }
    
    # Prepare task history info
    history = []
    for task in task_history:
        task_info = {
            "id": task.get("id"),
            "type": task.get("type"),
            "status": task.get("status"),
            "file_name": task.get("file_name") if "file_name" in task else None,
            "target_language": task.get("target_language") if "target_language" in task else None,
            "timestamp": task.get("timestamp")
        }
        
        # Add success counts for completed tasks
        if task.get("status") == "completed":
            if task.get("type") == "ingest":
                task_info["decompiled_count"] = len(task.get("decompiled_files", []))
                task_info["spec_count"] = len(task.get("spec_files", []))
                task_info["reconstructed_count"] = len(task.get("reconstructed_files", []))
            elif task.get("type") == "mimic":
                task_info["mimicked_count"] = len(task.get("mimicked_files", []))
                task_info["mimicked_dir"] = task.get("mimicked_dir")
        
        history.append(task_info)
    
    return jsonify({
        "current": current,
        "history": history
    })

@app.route('/task/<task_id>')
def get_task_details(task_id):
    """Get detailed information about a task"""
    # Find task in history or current task
    task = None
    if current_task and current_task.get("id") == task_id:
        task = current_task
    else:
        for t in task_history:
            if t.get("id") == task_id:
                task = t
                break
    
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    # Prepare response based on task type and status
    result = {
        "id": task.get("id"),
        "type": task.get("type"),
        "status": task.get("status"),
        "timestamp": task.get("timestamp")
    }
    
    if task.get("type") == "ingest":
        result["file_name"] = task.get("file_name")
        result["file_path"] = task.get("file_path")
        
        if task.get("status") == "completed":
            result["decompiled_files"] = [os.path.basename(f) for f in task.get("decompiled_files", [])]
            result["spec_files"] = [os.path.basename(f) for f in task.get("spec_files", [])]
            result["reconstructed_files"] = [os.path.basename(f) for f in task.get("reconstructed_files", [])]
    
    elif task.get("type") == "mimic":
        result["target_language"] = task.get("target_language")