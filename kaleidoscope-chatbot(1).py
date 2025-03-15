#!/usr/bin/env python3
"""
Kaleidoscope AI Chatbot
=======================
An open-source chatbot interface for the Kaleidoscope AI system that helps
users analyze, decompile, and mimic software through a conversational interface.

This chatbot uses the Kaleidoscope core engine to process software and provides
a friendly interface for interacting with the system.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Ensure the kaleidoscope_core module is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaleidoscope_core import KaleidoscopeCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KaleidoscopeChatbot:
    """Interactive chatbot for the Kaleidoscope AI system"""
    
    def __init__(self, work_dir: str = None):
        """
        Initialize the chatbot
        
        Args:
            work_dir: Working directory for the Kaleidoscope core
        """
        self.work_dir = work_dir or os.path.join(os.getcwd(), "kaleidoscope_workdir")
        self.kaleidoscope = KaleidoscopeCore(work_dir=self.work_dir)
        self.current_task = None
        self.task_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.session_history = []
        
        # Welcome message components
        self.welcome_message = [
            "Welcome to Kaleidoscope AI Chatbot!",
            "I can help you analyze, decompile, and mimic software through a conversational interface.",
            "Type 'help' to see available commands or 'exit' to quit."
        ]
        
        # Command handlers
        self.commands = {
            "help": self._handle_help,
            "ingest": self._handle_ingest,
            "status": self._handle_status,
            "list": self._handle_list,
            "mimic": self._handle_mimic,
            "info": self._handle_info,
            "clear": self._handle_clear,
            "exit": self._handle_exit
        }
    
    def start(self):
        """Start the chatbot"""
        self.running = True
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        # Print welcome message
        for line in self.welcome_message:
            print(line)
        print()
        
        # Main interaction loop
        while self.running:
            try:
                # Get user input
                user_input = input("> ").strip()
                
                # Process user input
                if not user_input:
                    continue
                
                # Record in history
                self.session_history.append({"role": "user", "content": user_input})
                
                # Process command
                self._process_input(user_input)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")
                print(f"Sorry, an error occurred: {str(e)}")
    
    def _process_input(self, user_input: str):
        """
        Process user input and execute appropriate command
        
        Args:
            user_input: User input string
        """
        # Split into command and arguments
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Check if command exists
        if command in self.commands:
            # Execute command handler
            response = self.commands[command](args)
            
            # Record in history
            self.session_history.append({"role": "assistant", "content": response})
            
            # Print response
            print(response)
        else:
            # Handle unknown command
            response = f"Unknown command: '{command}'. Type 'help' to see available commands."
            self.session_history.append({"role": "assistant", "content": response})
            print(response)
    
    def _worker_loop(self):
        """Background worker thread to process tasks"""
        while self.running:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1.0)
                
                # Process task
                if task["type"] == "ingest":
                    self._worker_ingest(task)
                elif task["type"] == "mimic":
                    self._worker_mimic(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks in queue
                pass
            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
                
                # Update current task status
                if self.current_task:
                    self.current_task["status"] = "error"
                    self.current_task["error"] = str(e)
                    self.current_task = None
    
    def _worker_ingest(self, task: Dict[str, Any]):
        """
        Worker function to ingest software
        
        Args:
            task: Task information
        """
        try:
            # Set as current task
            self.current_task = task
            self.current_task["status"] = "processing"
            
            # Ingest software
            file_path = task["file_path"]
            
            print(f"Ingesting {file_path}... This may take a while.")
            
            # Run ingestion
            result = self.kaleidoscope.ingest_software(file_path)
            
            # Update task with result
            task.update(result)
            task["status"] = result["status"]
            
            if result["status"] == "completed":
                print(f"\nIngestion completed successfully!")
                print(f"- Decompiled files: {len(result['decompiled_files'])}")
                print(f"- Specification files: {len(result['spec_files'])}")
                print(f"- Reconstructed files: {len(result['reconstructed_files'])}")
            else:
                print(f"\nIngestion failed: {result['status']}")
                if "error" in result:
                    print(f"Error: {result['error']}")
            
        except Exception as e:
            logger.error(f"Error in ingestion: {str(e)}")
            task["status"] = "error"
            task["error"] = str(e)
            print(f"\nError during ingestion: {str(e)}")
        finally:
            # Clear current task
            self.current_task = None
    
    def _worker_mimic(self, task: Dict[str, Any]):
        """
        Worker function to mimic software
        
        Args:
            task: Task information
        """
        try:
            # Set as current task
            self.current_task = task
            self.current_task["status"] = "processing"
            
            # Mimic software
            spec_files = task["spec_files"]
            target_language = task["target_language"]
            
            print(f"Generating mimicked version in {target_language}... This may take a while.")
            
            # Run mimicry
            result = self.kaleidoscope.mimic_software(spec_files, target_language)
            
            # Update task with result
            task.update(result)
            task["status"] = result["status"]
            
            if result["status"] == "completed":
                print(f"\nMimicry completed successfully!")
                print(f"- Generated {len(result['mimicked_files'])} files")
                print(f"- Output directory: {result['mimicked_dir']}")
            else:
                print(f"\nMimicry failed: {result['status']}")
                if "error" in result:
                    print(f"Error: {result['error']}")
            
        except Exception as e:
            logger.error(f"Error in mimicry: {str(e)}")
            task["status"] = "error"
            task["error"] = str(e)
            print(f"\nError during mimicry: {str(e)}")
        finally:
            # Clear current task
            self.current_task = None
    
    def _handle_help(self, args: str) -> str:
        """
        Handle 'help' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        return """
Available commands:
  help                       Show this help message
  ingest <file_path>         Analyze and decompile a software file
  status                     Show status of current task
  list [decompiled|specs|reconstructed]  List files from most recent ingestion
  mimic <language>           Create a mimicked version in the specified language
  info                       Show information about the current session
  clear                      Clear the session history
  exit                       Exit the chatbot
        """.strip()
    
    def _handle_ingest(self, args: str) -> str:
        """
        Handle 'ingest' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        if not args:
            return "Please specify a file path to ingest. Usage: ingest <file_path>"
        
        file_path = args.strip()
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        # Create task
        task = {
            "type": "ingest",
            "file_path": file_path,
            "status": "queued",
            "timestamp": time.time()
        }
        
        # Add to queue
        self.task_queue.put(task)
        
        return f"Queued ingestion of {file_path}. Use 'status' to check progress."
    
    def _handle_status(self, args: str) -> str:
        """
        Handle 'status' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        if not self.current_task:
            return "No task is currently running."
        
        task_type = self.current_task["type"]
        task_status = self.current_task["status"]
        
        if task_type == "ingest":
            return f"Ingesting {self.current_task['file_path']} - Status: {task_status}"
        elif task_type == "mimic":
            return f"Mimicking in {self.current_task['target_language']} - Status: {task_status}"
        else:
            return f"Unknown task type: {task_type} - Status: {task_status}"
    
    def _handle_list(self, args: str) -> str:
        """
        Handle 'list' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        # Check if we have any results from ingestion
        if not hasattr(self, "current_task") or not self.current_task:
            for task in self.task_queue.queue:
                if task["type"] == "ingest" and task["status"] == "completed":
                    self.current_task = task
                    break
        
        if not self.current_task or "status" not in self.current_task or self.current_task["status"] != "completed":
            return "No completed ingestion results available. Run 'ingest <file_path>' first."
        
        # Determine which files to list
        category = args.strip().lower() if args else "all"
        
        if category == "decompiled" or category == "all":
            decompiled_files = self.current_task.get("decompiled_files", [])
            if not decompiled_files:
                return "No decompiled files available."
            
            response = "Decompiled files:\n"
            for i, file_path in enumerate(decompiled_files):
                response += f"{i+1}. {os.path.basename(file_path)}\n"
        
        if category == "specs" or category == "all":
            spec_files = self.current_task.get("spec_files", [])
            if not spec_files:
                return "No specification files available."
            
            if category == "all":
                response += "\n"
            else:
                response = ""
                
            response += "Specification files:\n"
            for i, file_path in enumerate(spec_files):
                response += f"{i+1}. {os.path.basename(file_path)}\n"
        
        if category == "reconstructed" or category == "all":
            reconstructed_files = self.current_task.get("reconstructed_files", [])
            if not reconstructed_files:
                return "No reconstructed files available."
            
            if category == "all":
                response += "\n"
            else:
                response = ""
                
            response += "Reconstructed files:\n"
            for i, file_path in enumerate(reconstructed_files):
                response += f"{i+1}. {os.path.basename(file_path)}\n"
        
        if category not in ["all", "decompiled", "specs", "reconstructed"]:
            return f"Unknown category: {category}. Use 'decompiled', 'specs', 'reconstructed', or leave blank for all."
        
        return response.strip()
    
    def _handle_mimic(self, args: str) -> str:
        """
        Handle 'mimic' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        if not args:
            return "Please specify a target language. Usage: mimic <language>"
        
        target_language = args.strip().lower()
        
        # Check if we have specification files
        if not hasattr(self, "current_task") or not self.current_task or "spec_files" not in self.current_task:
            return "No specification files available. Run 'ingest <file_path>' first."
        
        spec_files = self.current_task.get("spec_files", [])
        if not spec_files:
            return "No specification files available. Run 'ingest <file_path>' first."
        
        # Validate target language
        valid_languages = ["python", "javascript", "c", "cpp", "c++", "java"]
        if target_language not in valid_languages:
            return f"Unsupported language: {target_language}. Supported languages: {', '.join(valid_languages)}"
        
        # Map language aliases
        if target_language in ["c++", "cpp"]:
            target_language = "cpp"
        
        # Create task
        task = {
            "type": "mimic",
            "spec_files": spec_files,
            "target_language": target_language,
            "status": "queued",
            "timestamp": time.time()
        }
        
        # Add to queue
        self.task_queue.put(task)
        
        return f"Queued mimicry in {target_language}. Use 'status' to check progress."
    
    def _handle_info(self, args: str) -> str:
        """
        Handle 'info' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        # Collect system information
        info = [
            f"Kaleidoscope AI Chatbot",
            f"Working directory: {self.work_dir}",
            f"Session commands: {len([h for h in self.session_history if h['role'] == 'user'])}",
            f"Ingestion tasks: {sum(1 for t in self.task_queue.queue if t['type'] == 'ingest')}",
            f"Mimicry tasks: {sum(1 for t in self.task_queue.queue if t['type'] == 'mimic')}"
        ]
        
        # Add information about current task if available
        if self.current_task:
            info.append(f"Current task: {self.current_task['type']} - {self.current_task['status']}")
        
        return "\n".join(info)
    
    def _handle_clear(self, args: str) -> str:
        """
        Handle 'clear' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        self.session_history = []
        return "Session history cleared."
    
    def _handle_exit(self, args: str) -> str:
        """
        Handle 'exit' command
        
        Args:
            args: Command arguments
            
        Returns:
            Response message
        """
        self.running = False
        return "Goodbye! Exiting Kaleidoscope AI Chatbot."

def main():
    """Main entry point for the chatbot"""
    parser = argparse.ArgumentParser(description="Kaleidoscope AI Chatbot")
    parser.add_argument("--work-dir", "-w", help="Working directory", default=None)
    
    args = parser.parse_args()
    
    try:
        # Create and start chatbot
        chatbot = KaleidoscopeChatbot(work_dir=args.work_dir)
        chatbot.start()
    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
