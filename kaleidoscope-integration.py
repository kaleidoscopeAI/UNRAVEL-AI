#!/usr/bin/env python3
"""
Kaleidoscope AI System Integration
Main entry point for launching and integrating all Kaleidoscope components.
"""

import os
import sys
import time
import json
import argparse
import logging
import threading
import multiprocessing as mp
import signal
import subprocess
from typing import Dict, List, Any, Optional, Union
import atexit

# Import Kaleidoscope components
from kaleidoscope_core import KaleidoscopeSystem
from kaleidoscope_extensions import KaleidoscopeExtensions, SoftwareIngestion, DrugDiscovery, PatternRecognition
from kaleidoscope_api import start_api
from kaleidoscope_visualization import KaleidoscopeVisualizer
from kaleidoscope_core import KaleidoscopeCore
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("kaleidoscope.log"), logging.StreamHandler()]
)
logger = logging.getLogger("KaleidoscopeIntegration")

class ModuleIntegration:
    """Handles integration and communication between modules"""
    
    def __init__(self, core: KaleidoscopeCore):
        self.core = core
        self.message_queue = asyncio.Queue()
        self.routes = {}
        
    async def route_message(self, source: str, target: str, message: Any):
        """Route a message from source to target module"""
        if target not in self.core.modules:
            raise ValueError(f"Target module not found: {target}")
            
        await self.message_queue.put({
            "source": source,
            "target": target,
            "message": message
        })

class KaleidoscopeManager:
    """
    Manager class for integrating and controlling all Kaleidoscope components.
    """
    
    def __init__(self, 
                config_path: str = None, 
                data_dir: str = './kaleidoscope_data',
                api_host: str = '0.0.0.0',
                api_port: int = 8000):
        """Initialize the Kaleidoscope manager."""
        self.config_path = config_path
        self.data_dir = data_dir
        self.api_host = api_host
        self.api_port = api_port
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize components
        self.system = None
        self.extensions = None
        self.visualizer = None
        
        # API server process
        self.api_process = None
        
        # System monitor thread
        self.monitor_thread = None
        self.monitor_stop_flag = threading.Event()
        
        # Register cleanup handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Kaleidoscope Manager initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        default_config = {
            'system': {
                'db_path': os.path.join(self.data_dir, 'kaleidoscope.db'),
                'initial_nodes': 10,
                'initial_supernodes': 3,
                'cube_dimensions': (10, 10, 10)
            },
            'extensions': {
                'use_ghidra': False,
                'use_retdec': True,
                'use_radare2': True,
                'use_remote_databases': True
            },
            'api': {
                'host': self.api_host,
                'port': self.api_port,
                'debug': False
            },
            'visualization': {
                'output_dir': os.path.join(self.data_dir, 'visualizations')
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge configurations
                self._merge_configs(default_config, user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        return default_config
    
    def _merge_configs(self, target: Dict, source: Dict) -> None:
        """Recursively merge source config into target config."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_configs(target[key], value)
            else:
                target[key] = value
    
    def start(self) -> None:
        """Start all Kaleidoscope components."""
        try:
            # Start core system
            logger.info("Starting Kaleidoscope core system...")
            self.system = KaleidoscopeSystem(config_path=self.config_path)
            
            # Start extensions
            logger.info("Initializing Kaleidoscope extensions...")
            self.extensions = KaleidoscopeExtensions(
                working_dir=self.data_dir,
                config=self.config['extensions']
            )
            
            # Initialize visualizer
            logger.info("Initializing visualizer...")
            self.visualizer = KaleidoscopeVisualizer(
                output_dir=self.config['visualization']['output_dir']
            )
            
            # Start API server
            logger.info("Starting API server...")
            self._start_api_server()
            
            # Start system monitor
            logger.info("Starting system monitor...")
            self._start_system_monitor()
            
            logger.info("Kaleidoscope system started successfully")
        except Exception as e:
            logger.error(f"Error starting Kaleidoscope system: {str(e)}")
            self.shutdown()
            raise
    
    def _start_api_server(self) -> None:
        """Start the API server in a separate process."""
        api_config = self.config['api']
        
        # Start the API server in a separate process
        self.api_process = mp.Process(
            target=start_api,
            kwargs={
                'host': api_config['host'],
                'port': api_config['port']
            }
        )
        self.api_process.start()
        
        # Wait a moment for the server to start
        time.sleep(1)
        
        if not self.api_process.is_alive():
            raise RuntimeError("Failed to start API server")
        
        logger.info(f"API server started on {api_config['host']}:{api_config['port']}")
    
    def _start_system_monitor(self) -> None:
        """Start the system monitor thread."""
        self.monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _system_monitor_loop(self) -> None:
        """System monitor loop that periodically checks the system status and performs maintenance."""
        logger.info("System monitor started")
        
        # Evolution interval (seconds)
        evolution_interval = 300  # 5 minutes
        last_evolution = time.time()
        
        while not self.monitor_stop_flag.is_set():
            try:
                # Get system status
                status = self.system.get_system_status()
                
                # Log basic status
                logger.info(f"System status: Nodes={status['nodes']['active']}/{status['nodes']['total']}, "
                          f"SuperNodes={status['supernodes']['active']}/{status['supernodes']['total']}, "
                          f"Insights={status['insights_generated']}")
                
                # Check if it's time for system evolution
                current_time = time.time()
                if current_time - last_evolution > evolution_interval:
                    logger.info("Performing system evolution...")
                    
                    # Regenerate node energy
                    self.system.regenerate_nodes()
                    
                    # Evolve system
                    evolution_result = self.system.evolve_system()
                    
                    logger.info(f"Evolution completed: {evolution_result['new_pathways']} new pathways, "
                              f"{evolution_result['pruned_pathways']} pruned pathways")
                    
                    last_evolution = current_time
                
                # Short sleep to prevent high CPU usage
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in system monitor: {str(e)}")
                time.sleep(60)  # Longer sleep on error
    
    def process_input(self, data: Any, data_type: str = None) -> Dict:
        """Process input data through the Kaleidoscope system."""
        if not self.system:
            raise RuntimeError("Kaleidoscope system not started")
        
        try:
            # Process through system
            result = self.system.process_input(data, data_type)
            return result
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_binary(self, binary_path: str, output_language: str = "c") -> Dict:
        """Analyze a binary file using the SoftwareIngestion extension."""
        if not self.extensions:
            raise RuntimeError("Kaleidoscope extensions not initialized")
        
        try:
            # Decompile binary
            result = self.extensions.decompile_binary(
                binary_path=binary_path,
                output_language=output_language,
                analyze=True
            )
            
            if result.get('success'):
                # Detect patterns
                binary_hash = result.get('binary_hash')
                pattern_result = self.extensions.detect_code_patterns(binary_hash)
                
                # Combine results
                result['pattern_analysis'] = pattern_result
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing binary: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_molecule(self, molecule: str, input_format: str = "smiles") -> Dict:
        """Analyze a molecule using the DrugDiscovery extension."""
        if not self.extensions:
            raise RuntimeError("Kaleidoscope extensions not initialized")
        
        try:
            # Process molecule
            result = self.extensions.analyze_molecule(
                molecule=molecule,
                input_format=input_format
            )
            
            if result.get('success'):
                # Generate visualization if possible
                if self.visualizer and 'molecule_id' in result:
                    try:
                        viz_result = self.visualizer.visualize_molecular_structure(
                            result,
                            output_file=f"molecule_{result['molecule_id'][:8]}.html"
                        )
                        if viz_result:
                            result['visualization'] = viz_result
                    except Exception as viz_error:
                        logger.warning(f"Visualization error: {str(viz_error)}")
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing molecule: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def detect_patterns(self, data: Any, data_type: str = None) -> Dict:
        """Detect patterns in data using the PatternRecognition extension."""
        if not self.extensions:
            raise RuntimeError("Kaleidoscope extensions not initialized")
        
        try:
            # Analyze patterns
            result = self.extensions.analyze_patterns(
                data=data,
                data_type=data_type
            )
            
            if result.get('success'):
                # Generate visualization if possible
                if self.visualizer and 'patterns' in result:
                    try:
                        viz_result = self.visualizer.visualize_pattern_analysis(
                            result,
                            output_file=f"patterns_{int(time.time())}.html"
                        )
                        if viz_result:
                            result['visualization'] = viz_result
                    except Exception as viz_error:
                        logger.warning(f"Visualization error: {str(viz_error)}")
            
            return result
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def visualize_cube(self, output_file: str = None, interactive: bool = True) -> str:
        """Generate visualization of the Kaleidoscope cube structure."""
        if not self.system or not self.visualizer:
            raise RuntimeError("Kaleidoscope system or visualizer not initialized")
        
        try:
            # Get cube data
            cube_data = self.system.cube.get_visualization_data()
            
            # Generate visualization
            return self.visualizer.visualize_cube(
                cube_data=cube_data,
                output_file=output_file,
                interactive=interactive
            )
        except Exception as e:
            logger.error(f"Error visualizing cube: {str(e)}")
            return None
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def shutdown(self) -> None:
        """Shutdown all Kaleidoscope components."""
        logger.info("Shutting down Kaleidoscope system...")
        
        # Stop monitor thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.info("Stopping system monitor...")
            self.monitor_stop_flag.set()
            self.monitor_thread.join(timeout=5)
        
        # Stop API server
        if self.api_process and self.api_process.is_alive():
            logger.info("Stopping API server...")
            self.api_process.terminate()
            self.api_process.join(timeout=5)
            
            # Force kill if still alive
            if self.api_process.is_alive():
                self.api_process.kill()
        
        # Shutdown core system
        if self.system:
            logger.info("Shutting down core system...")
            self.system.shutdown()
        
        logger.info("Kaleidoscope system shutdown complete")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Kaleidoscope AI System")
    
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data-dir", type=str, default="./kaleidoscope_data",
        help="Data directory path"
    )
    
    parser.add_argument(
        "--api-host", type=str, default="0.0.0.0",
        help="API server host"
    )
    
    parser.add_argument(
        "--api-port", type=int, default=8000,
        help="API server port"
    )
    
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with sample data"
    )
    
    return parser.parse_args()


def run_demo(manager: KaleidoscopeManager) -> None:
    """Run a demo with sample data."""
    logger.info("Running Kaleidoscope demo...")
    
    # 1. Process text data
    logger.info("Processing text data...")
    text_data = """
    The novel drug compound XJ-42 shows promising results in initial trials.
    It binds to the ACE2 receptor with high affinity (Kd = 5.2 nM) and
    demonstrates significant antiviral activity against SARS-CoV-2.
    The chemical structure contains a modified benzothiazole core with
    three fluorine atoms at positions 2, 4, and 6.
    """
    
    text_result = manager.process_input(text_data, "text")
    logger.info(f"Text processing result: {text_result['success']}, "
              f"Insights: {len(text_result.get('insights_generated', []))}")
    
    # 2. Analyze molecule
    logger.info("Analyzing molecule...")
    # Sample SMILES for aspirin
    molecule = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    molecule_result = manager.analyze_molecule(molecule)
    logger.info(f"Molecule analysis result: {molecule_result['success']}")
    
    # 3. Detect patterns
    logger.info("Detecting patterns...")
    code_sample = """
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def process_data(data):
        result = bubble_sort(data)
        avg = sum(result) / len(result)
        return {"sorted": result, "average": avg}
    """
    
    pattern_result = manager.detect_patterns(code_sample, "code")
    logger.info(f"Pattern detection result: {pattern_result['success']}, "
              f"Patterns: {pattern_result.get('patterns_count', 0)}")
    
    # 4. Visualize cube
    logger.info("Generating cube visualization...")
    viz_result = manager.visualize_cube("demo_cube.html")
    if viz_result:
        logger.info(f"Visualization saved to {viz_result}")
    
    logger.info("Demo completed")


def main():
    """Main entry point for the Kaleidoscope system."""
    # Parse arguments
    args = parse_arguments()
    
    # Create and start manager
    manager = KaleidoscopeManager(
        config_path=args.config,
        data_dir=args.data_dir,
        api_host=args.api_host,
        api_port=args.api_port
    )
    
    try:
        # Start system
        manager.start()
        
        # Run demo if requested
        if args.demo:
            run_demo(manager)
        
        # Keep main thread running
        logger.info("Kaleidoscope system is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
