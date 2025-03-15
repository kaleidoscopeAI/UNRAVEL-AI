#!/usr/bin/env python3
"""
Kaleidoscope AI System Client
Python client for interacting with the Kaleidoscope AI system API.
"""

import requests
import json
import time
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union, BinaryIO
import argparse
from kaleidoscope-core import KaleidoscopeCore
from kaleidoscope-integration import ModuleIntegration
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("KaleidoscopeClient")

class KaleidoscopeClient:
    """
    Client for interacting with the Kaleidoscope AI system API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.last_response = None
        
        logger.info(f"Kaleidoscope client initialized with base URL: {self.base_url}")
    
    def _make_request(self, 
                    method: str, 
                    endpoint: str, 
                    data: Dict = None, 
                    files: Dict = None, 
                    params: Dict = None) -> Dict:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params)
            elif method.upper() == 'POST':
                if files:
                    response = self.session.post(url, data=data, files=files, params=params)
                else:
                    headers = {'Content-Type': 'application/json'}
                    response = self.session.post(url, json=data, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            self.last_response = response
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    logger.error(f"API error response: {error_data}")
                    return error_data
                except:
                    logger.error(f"API error response (text): {e.response.text}")
            
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "timestamp": time.time()
            }
    
    # Core API methods
    def get_status(self) -> Dict:
        """Get the current system status."""
        return self._make_request('GET', 'status')
    
    def process_data(self, 
                   data: Any, 
                   data_type: str = None, 
                   sync: bool = False) -> Dict:
        """
        Process data through the Kaleidoscope AI system.
        
        Args:
            data: The data to process
            data_type: Optional type of the data
            sync: Whether to process synchronously (True) or asynchronously (False)
        
        Returns:
            API response dictionary
        """
        endpoint = 'process/sync' if sync else 'process'
        
        return self._make_request('POST', endpoint, data={
            'data': data,
            'data_type': data_type
        })
    
    def evolve_system(self) -> Dict:
        """Trigger system evolution."""
        return self._make_request('POST', 'evolve')
    
    # Software Ingestion methods
    def decompile_binary(self, 
                       binary_path: str, 
                       output_language: str = "c", 
                       analyze: bool = True) -> Dict:
        """
        Decompile a binary file to the specified language.
        
        Args:
            binary_path: Path to the binary file
            output_language: Target language for decompilation
            analyze: Whether to perform analysis
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', 'extensions/software/decompile', data={
            'binary_path': binary_path,
            'output_language': output_language,
            'analyze': analyze
        })
    
    def detect_code_patterns(self, binary_hash: str) -> Dict:
        """
        Detect patterns in decompiled code.
        
        Args:
            binary_hash: Hash of the binary file
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', f'extensions/software/patterns/{binary_hash}')
    
    def reconstruct_software(self, 
                          binary_hash: str, 
                          language: str = "python", 
                          enhance: bool = True) -> Dict:
        """
        Reconstruct software from binary with enhancements.
        
        Args:
            binary_hash: Hash of the binary file
            language: Target language for reconstruction
            enhance: Whether to apply enhancements
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', f'extensions/software/reconstruct/{binary_hash}', 
                               params={'language': language, 'enhance': enhance})
    
    def upload_binary(self, file_path: str) -> Dict:
        """
        Upload a binary file for analysis.
        
        Args:
            file_path: Path to the binary file
        
        Returns:
            API response dictionary
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "timestamp": time.time()
            }
        
        with open(file_path, 'rb') as f:
            return self._make_request('POST', 'upload/binary', 
                                   files={'file': (os.path.basename(file_path), f)})
    
    # Drug Discovery methods
    def analyze_molecule(self, 
                       molecule: str, 
                       input_format: str = "smiles") -> Dict:
        """
        Analyze a molecule's properties.
        
        Args:
            molecule: Molecule representation (e.g., SMILES string)
            input_format: Format of the molecule representation
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', 'extensions/drug/analyze', data={
            'molecule': molecule,
            'input_format': input_format
        })
    
    def generate_molecular_variants(self, 
                                 molecule_id: str, 
                                 count: int = 5) -> Dict:
        """
        Generate variants of a molecule.
        
        Args:
            molecule_id: ID of the molecule
            count: Number of variants to generate
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', f'extensions/drug/variants/{molecule_id}', 
                               params={'count': count})
    
    def simulate_docking(self, 
                       molecule_id: str, 
                       target: str, 
                       exhaustiveness: int = 8) -> Dict:
        """
        Simulate molecular docking with a target protein.
        
        Args:
            molecule_id: ID of the molecule
            target: Target protein
            exhaustiveness: Docking exhaustiveness level
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', 'extensions/drug/docking', data={
            'molecule_id': molecule_id,
            'target': target,
            'exhaustiveness': exhaustiveness
        })
    
    def predict_interactions(self, 
                          molecule_id: str, 
                          against_drugs: List[str] = None) -> Dict:
        """
        Predict drug interactions.
        
        Args:
            molecule_id: ID of the molecule
            against_drugs: List of drug IDs to check against
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', f'extensions/drug/interactions/{molecule_id}', 
                               params={'against_drugs': against_drugs})
    
    def upload_molecule(self, 
                      file_path: str, 
                      format: str = "sdf") -> Dict:
        """
        Upload a molecule file for analysis.
        
        Args:
            file_path: Path to the molecule file
            format: Format of the molecule file
        
        Returns:
            API response dictionary
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "timestamp": time.time()
            }
        
        with open(file_path, 'rb') as f:
            return self._make_request('POST', 'upload/molecule', 
                                   files={'file': (os.path.basename(file_path), f)},
                                   data={'format': format})
    
    # Pattern Recognition methods
    def analyze_patterns(self, 
                       data: Any, 
                       data_type: str = None) -> Dict:
        """
        Detect patterns in data.
        
        Args:
            data: The data to analyze
            data_type: Optional type of the data
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', 'extensions/patterns/analyze', data={
            'data': data,
            'data_type': data_type
        })
    
    def find_cross_domain_insights(self, domains: List[str]) -> Dict:
        """
        Find patterns across multiple domains.
        
        Args:
            domains: List of domains to analyze
        
        Returns:
            API response dictionary
        """
        return self._make_request('POST', 'extensions/patterns/cross-domain', data={
            'domains': domains
        })
    
    # Visualization methods
    def get_cube_visualization(self) -> Dict:
        """Get data for cube visualization."""
        return self._make_request('GET', 'visualize/cube')
    
    def get_network_visualization(self) -> Dict:
        """Get data for network visualization."""
        return self._make_request('GET', 'visualize/network')
    
    # Utility methods
    def pretty_print_response(self, response: Dict = None) -> None:
        """Pretty print the API response."""
        if response is None:
            response = self.last_response
        
        if response is None:
            print("No response to display")
            return
        
        print(json.dumps(response, indent=2))


def main():
    """Command-line interface for the Kaleidoscope client."""
    parser = argparse.ArgumentParser(description="Kaleidoscope AI System Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                      help="API base URL")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get system status")
    
    # Process data command
    process_parser = subparsers.add_parser("process", help="Process data")
    process_parser.add_argument("--data", type=str, required=True, 
                              help="Data to process (string or file path)")
    process_parser.add_argument("--data-type", type=str, help="Data type")
    process_parser.add_argument("--sync", action="store_true", 
                              help="Process synchronously")
    process_parser.add_argument("--from-file", action="store_true", 
                              help="Load data from file")
    
    # Evolve system command
    evolve_parser = subparsers.add_parser("evolve", help="Trigger system evolution")
    
    # Software ingestion commands
    decompile_parser = subparsers.add_parser("decompile", help="Decompile binary")
    decompile_parser.add_argument("--binary", type=str, required=True, 
                                help="Path to binary file")
    decompile_parser.add_argument("--language", type=str, default="c", 
                                help="Output language")
    decompile_parser.add_argument("--no-analyze", action="store_true", 
                                help="Skip analysis")
    
    patterns_parser = subparsers.add_parser("code-patterns", 
                                          help="Detect code patterns")
    patterns_parser.add_argument("--hash", type=str, required=True, 
                               help="Binary hash")
    
    reconstruct_parser = subparsers.add_parser("reconstruct", 
                                             help="Reconstruct software")
    reconstruct_parser.add_argument("--hash", type=str, required=True, 
                                  help="Binary hash")
    reconstruct_parser.add_argument("--language", type=str, default="python", 
                                  help="Target language")
    reconstruct_parser.add_argument("--no-enhance", action="store_true", 
                                  help="Skip enhancements")
    
    upload_binary_parser = subparsers.add_parser("upload-binary", 
                                               help="Upload binary file")
    upload_binary_parser.add_argument("--file", type=str, required=True, 
                                    help="Path to binary file")
    
    # Drug discovery commands
    molecule_parser = subparsers.add_parser("analyze-molecule", 
                                          help="Analyze molecule")
    molecule_parser.add_argument("--molecule", type=str, required=True, 
                               help="Molecule representation")
    molecule_parser.add_argument("--format", type=str, default="smiles", 
                               help="Input format")
    
    variants_parser = subparsers.add_parser("molecular-variants", 
                                          help="Generate molecular variants")
    variants_parser.add_argument("--id", type=str, required=True, 
                               help="Molecule ID")
    variants_parser.add_argument("--count", type=int, default=5, 
                               help="Number of variants")
    
    docking_parser = subparsers.add_parser("simulate-docking", 
                                         help="Simulate molecular docking")
    docking_parser.add_argument("--id", type=str, required=True, 
                              help="Molecule ID")
    docking_parser.add_argument("--target", type=str, required=True, 
                              help="Target protein")
    docking_parser.add_argument("--exhaustiveness", type=int, default=8, 
                              help="Docking exhaustiveness")
    
    interactions_parser = subparsers.add_parser("predict-interactions", 
                                              help="Predict drug interactions")
    interactions_parser.add_argument("--id", type=str, required=True, 
                                   help="Molecule ID")
    interactions_parser.add_argument("--against", type=str, nargs="+", 
                                   help="Drug IDs to check against")
    
    upload_molecule_parser = subparsers.add_parser("upload-molecule", 
                                                 help="Upload molecule file")
    upload_molecule_parser.add_argument("--file", type=str, required=True, 
                                      help="Path to molecule file")
    upload_molecule_parser.add_argument("--format", type=str, default="sdf", 
                                      help="Molecule file format")
    
    # Pattern recognition commands
    analyze_patterns_parser = subparsers.add_parser("analyze-patterns", 
                                                  help="Analyze patterns in data")
    analyze_patterns_parser.add_argument("--data", type=str, required=True, 
                                       help="Data to analyze (string or file path)")
    analyze_patterns_parser.add_argument("--data-type", type=str, 
                                       help="Data type")
    analyze_patterns_parser.add_argument("--from-file", action="store_true", 
                                       help="Load data from file")
    
    cross_domain_parser = subparsers.add_parser("cross-domain", 
                                              help="Find cross-domain insights")
    cross_domain_parser.add_argument("--domains", type=str, nargs="+", required=True, 
                                   help="Domains to analyze")
    
    # Visualization commands
    cube_viz_parser = subparsers.add_parser("cube-viz", 
                                          help="Get cube visualization data")
    
    network_viz_parser = subparsers.add_parser("network-viz", 
                                             help="Get network visualization data")
    
    args = parser.parse_args()
    
    # Initialize client
    client = KaleidoscopeClient(base_url=args.url)
    
    if args.command == "status":
        response = client.get_status()
    
    elif args.command == "process":
        data = args.data
        if args.from_file:
            with open(args.data, 'r') as f:
                data = f.read()
        response = client.process_data(data, args.data_type, args.sync)
    
    elif args.command == "evolve":
        response = client.evolve_system()
    
    elif args.command == "decompile":
        response = client.decompile_binary(
            args.binary, 
            args.language, 
            not args.no_analyze
        )
    
    elif args.command == "code-patterns":
        response = client.detect_code_patterns(args.hash)
    
    elif args.command == "reconstruct":
        response = client.reconstruct_software(
            args.hash, 
            args.language, 
            not args.no_enhance
        )
    
    elif args.command == "upload-binary":
        response = client.upload_binary(args.file)
    
    elif args.command == "analyze-molecule":
        response = client.analyze_molecule(args.molecule, args.format)
    
    elif args.command == "molecular-variants":
        response = client.generate_molecular_variants(args.id, args.count)
    
    elif args.command == "simulate-docking":
        response = client.simulate_docking(
            args.id, 
            args.target, 
            args.exhaustiveness
        )
    
    elif args.command == "predict-interactions":
        response = client.predict_interactions(args.id, args.against)
    
    elif args.command == "upload-molecule":
        response = client.upload_molecule(args.file, args.format)
    
    elif args.command == "analyze-patterns":
        data = args.data
        if args.from_file:
            with open(args.data, 'r') as f:
                data = f.read()
        response = client.analyze_patterns(data, args.data_type)
    
    elif args.command == "cross-domain":
        response = client.find_cross_domain_insights(args.domains)
    
    elif args.command == "cube-viz":
        response = client.get_cube_visualization()
    
    elif args.command == "network-viz":
        response = client.get_network_visualization()
    
    else:
        parser.print_help()
        return
    
    # Print response
    client.pretty_print_response(response)


if __name__ == "__main__":
    main()
