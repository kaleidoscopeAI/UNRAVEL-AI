#!/usr/bin/env python3
"""
Kaleidoscope AI System Setup
Installation and configuration script for the Kaleidoscope AI system.
"""

import os
import sys
import json
import argparse
import subprocess
import platform
import logging
import shutil
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("setup.log"), logging.StreamHandler()]
)
logger = logging.getLogger("KaleidoscopeSetup")

# Define dependencies
BASE_DEPENDENCIES = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "networkx>=2.6.0",
    "matplotlib>=3.4.0",
    "pandas>=1.3.0",
    "fastapi>=0.68.0",
    "uvicorn>=0.15.0",
    "plotly>=5.3.0",
    "requests>=2.26.0",
    "pydantic>=1.8.0",
    "python-multipart>=0.0.5"
]

OPTIONAL_DEPENDENCIES = {
    "rdkit": ["rdkit>=2021.03.1"],
    "decompilers": ["r2pipe>=1.6.0"],  # For radare2 integration
    "visualization": ["dash>=2.0.0", "kaleido>=0.2.1"]
}

class KaleidoscopeSetup:
    """
    Setup helper for the Kaleidoscope AI system.
    """
    
    def __init__(self, 
                install_dir: str = "./kaleidoscope",
                data_dir: str = "./kaleidoscope_data",
                config_path: str = "./kaleidoscope_config.json"):
        """Initialize the setup helper."""
        self.install_dir = os.path.abspath(install_dir)
        self.data_dir = os.path.abspath(data_dir)
        self.config_path = os.path.abspath(config_path)
        
        # Default configuration
        self.config = {
            'system': {
                'db_path': os.path.join(self.data_dir, 'kaleidoscope.db'),
                'initial_nodes': 10,
                'initial_supernodes': 3,
                'cube_dimensions': [10, 10, 10]
            },
            'extensions': {
                'use_ghidra': False,
                'use_retdec': False,
                'use_radare2': False,
                'use_remote_databases': True
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            },
            'visualization': {
                'output_dir': os.path.join(self.data_dir, 'visualizations')
            }
        }
        
        logger.info(f"Setup initialized with installation directory: {self.install_dir}")
    
    def check_system_requirements(self) -> bool:
        """Check if the system meets the requirements."""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_version = platform.python_version_tuple()
        python_version_str = ".".join(python_version)
        
        if int(python_version[0]) < 3 or (int(python_version[0]) == 3 and int(python_version[1]) < 8):
            logger.error(f"Python 3.8+ required, found {python_version_str}")
            return False
        
        logger.info(f"Python version: {python_version_str} - OK")
        
        # Check available disk space (>= 500MB)
        try:
            if os.name == 'posix':
                # UNIX-like systems
                disk_info = os.statvfs(self.install_dir)
                free_space_mb = (disk_info.f_bavail * disk_info.f_frsize) / (1024 * 1024)
            else:
                # Windows
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(self.install_dir), None, None, 
                    ctypes.pointer(free_bytes)
                )
                free_space_mb = free_bytes.value / (1024 * 1024)
            
            if free_space_mb < 500:
                logger.warning(f"Low disk space: {free_space_mb:.2f} MB")
                return False
            
            logger.info(f"Disk space: {free_space_mb:.2f} MB - OK")
        except Exception as e:
            logger.warning(f"Could not check disk space: {str(e)}")
        
        return True
    
    def install_dependencies(self, extras: List[str] = None) -> bool:
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        
        # Prepare dependencies list
        dependencies = BASE_DEPENDENCIES.copy()
        
        # Add optional dependencies if requested
        if extras:
            for extra in extras:
                if extra in OPTIONAL_DEPENDENCIES:
                    dependencies.extend(OPTIONAL_DEPENDENCIES[extra])
        
        # Install dependencies using pip
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + dependencies
            )
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {str(e)}")
            return False
    
    def check_external_tools(self) -> Dict[str, bool]:
        """Check for availability of external tools."""
        logger.info("Checking external tools...")
        
        tools = {
            'radare2': False,
            'retdec': False,
            'ghidra': False
        }
        
        # Check for radare2
        try:
            result = subprocess.run(['r2', '-v'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 timeout=2)
            tools['radare2'] = result.returncode == 0
            if tools['radare2']:
                logger.info("radare2 found")
            else:
                logger.info("radare2 not found")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.info("radare2 not found")
        
        # Check for RetDec
        retdec_path = os.environ.get('RETDEC_PATH')
        if retdec_path and os.path.exists(retdec_path):
            tools['retdec'] = True
            logger.info(f"RetDec found at {retdec_path}")
        else:
            logger.info("RetDec not found")
        
        # Check for Ghidra
        ghidra_path = os.environ.get('GHIDRA_PATH')
        if ghidra_path and os.path.exists(ghidra_path):
            tools['ghidra'] = True
            logger.info(f"Ghidra found at {ghidra_path}")
        else:
            logger.info("Ghidra not found")
        
        # Update config with available tools
        self.config['extensions']['use_radare2'] = tools['radare2']
        self.config['extensions']['use_retdec'] = tools['retdec']
        self.config['extensions']['use_ghidra'] = tools['ghidra']
        
        return tools
    
    def setup_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("Setting up directories...")
        
        try:
            # Create installation directory
            os.makedirs(self.install_dir, exist_ok=True)
            logger.info(f"Created installation directory: {self.install_dir}")
            
            # Create data directory
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
            
            # Create sub-directories
            dirs = [
                os.path.join(self.data_dir, 'software_analysis'),
                os.path.join(self.data_dir, 'drug_discovery'),
                os.path.join(self.data_dir, 'pattern_recognition'),
                os.path.join(self.data_dir, 'visualizations'),
                os.path.join(self.data_dir, 'uploads')
            ]
            
            for d in dirs:
                os.makedirs(d, exist_ok=True)
                logger.info(f"Created directory: {d}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            return False
    
    def create_config_file(self) -> bool:
        """Create configuration file."""
        logger.info(f"Creating configuration file: {self.config_path}")
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logger.info("Configuration file created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating configuration file: {str(e)}")
            return False
    
    def check_rdkit_installation(self) -> bool:
        """Check if RDKit is properly installed."""
        logger.info("Checking RDKit installation...")
        
        try:
            import rdkit
            from rdkit import Chem
            
            # Simple test
            mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # Aspirin
            if mol is None:
                logger.warning("RDKit installation may have issues")
                return False
            
            logger.info("RDKit is properly installed")
            return True
        except ImportError:
            logger.warning("RDKit is not installed")
            return False
        except Exception as e:
            logger.error(f"Error checking RDKit: {str(e)}")
            return False
    
    def copy_source_files(self, source_dir: str) -> bool:
        """Copy source files to installation directory."""
        logger.info(f"Copying source files from {source_dir} to {self.install_dir}...")
        
        try:
            # Copy Python files
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.endswith('.py'):
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, source_dir)
                        dst_path = os.path.join(self.install_dir, rel_path)
                        
                        # Create destination directory if it doesn't exist
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        
                        # Copy file
                        shutil.copy2(src_path, dst_path)
                        logger.info(f"Copied {rel_path}")
            
            # Create __init__.py files for packages
            for root, dirs, _ in os.walk(self.install_dir):
                for d in dirs:
                    init_path = os.path.join(root, d, '__init__.py')
                    if not os.path.exists(init_path):
                        with open(init_path, 'w') as f:
                            f.write(f'"""\n{d} package\n"""\n')
                        logger.info(f"Created {os.path.relpath(init_path, self.install_dir)}")
            
            return True
        except Exception as e:
            logger.error(f"Error copying source files: {str(e)}")
            return False
    
    def create_launcher(self) -> bool:
        """Create launcher script."""
        logger.info("Creating launcher script...")
        
        launcher_path = os.path.join(self.install_dir, 'launch_kaleidoscope.py')
        
        try:
            with open(launcher_path, 'w') as f:
                f.write(f"""#!/usr/bin/env python3
\"\"\"
Kaleidoscope AI System Launcher
\"\"\"

import os
import sys
import argparse

# Add installation directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaleidoscope_integration import main

if __name__ == "__main__":
    main()
""")
            
            # Make launcher executable on Unix-like systems
            if os.name == 'posix':
                os.chmod(launcher_path, 0o755)
            
            logger.info(f"Launcher script created: {launcher_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating launcher script: {str(e)}")
            return False
    
    def run_setup(self, source_dir: str = None, extras: List[str] = None) -> bool:
        """Run the complete setup process."""
        logger.info("Running Kaleidoscope AI setup...")
        
        # Check system requirements
        if not self.check_system_requirements():
            logger.error("System does not meet requirements")
            return False
        
        # Install dependencies
        if not self.install_dependencies(extras):
            logger.error("Failed to install dependencies")
            return False
        
        # Check external tools
        self.check_external_tools()
        
        # Create directories
        if not self.setup_directories():
            logger.error("Failed to set up directories")
            return False
        
        # Create configuration file
        if not self.create_config_file():
            logger.error("Failed to create configuration file")
            return False
        
        # Copy source files if provided
        if source_dir:
            if not self.copy_source_files(source_dir):
                logger.error("Failed to copy source files")
                return False
            
            # Create launcher
            if not self.create_launcher():
                logger.error("Failed to create launcher")
                return False
        
        # Check RDKit if drug discovery is enabled
        if extras and 'rdkit' in extras:
            self.check_rdkit_installation()
        
        logger.info("Kaleidoscope AI setup completed successfully")
        return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Kaleidoscope AI Setup")
    
    parser.add_argument(
        "--install-dir", type=str, default="./kaleidoscope",
        help="Installation directory"
    )
    
    parser.add_argument(
        "--data-dir", type=str, default="./kaleidoscope_data",
        help="Data directory"
    )
    
    parser.add_argument(
        "--config", type=str, default="./kaleidoscope_config.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help="Source directory (optional)"
    )
    
    parser.add_argument(
        "--extras", type=str, nargs="+", choices=["rdkit", "decompilers", "visualization"],
        help="Extra dependencies to install"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the setup script."""
    # Parse arguments
    args = parse_arguments()
    
    # Create setup helper
    setup = KaleidoscopeSetup(
        install_dir=args.install_dir,
        data_dir=args.data_dir,
        config_path=args.config
    )
    
    # Run setup
    if setup.run_setup(source_dir=args.source_dir, extras=args.extras):
        logger.info("Setup completed successfully")
        sys.exit(0)
    else:
        logger.error("Setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
