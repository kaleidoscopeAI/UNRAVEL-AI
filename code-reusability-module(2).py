#!/usr/bin/env python3
"""
Kaleidoscope AI - Code Reusability Architecture
===============================================
A comprehensive architecture for code reusability across different programming 
languages and analysis tasks in the Kaleidoscope AI system.

This module provides:
1. Unified AST representation for language-agnostic analysis
2. Language adapters for parsing and generating code
3. Plugin architecture for extending system capabilities
4. Common pipelines for code analysis and transformation
"""

import os
import sys
import importlib
import inspect
import pkgutil
import logging
import json
import asyncio
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, Type, TypeVar, Generic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_arch.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

###############################
# Unified AST Representation
###############################

class NodeType(Enum):
    """Types of nodes in the unified AST"""
    ROOT = auto()
    FILE = auto()
    MODULE = auto()
    CLASS = auto()
    FUNCTION = auto()
    METHOD = auto()
    INTERFACE = auto()
    ENUM = auto()
    VARIABLE = auto()
    CONSTANT = auto()
    IMPORT = auto()
    EXPRESSION = auto()
    STATEMENT = auto()
    PARAMETER = auto()
    RETURN = auto()
    CONDITION = auto()
    LOOP = auto()
    COMMENT = auto()
    ANNOTATION = auto()
    DECORATOR = auto()
    NAMESPACE = auto()
    PROPERTY = auto()
    OPERATOR = auto()
    LITERAL = auto()
    UNKNOWN = auto()

@dataclass
class Position:
    """Position in source code"""
    line: int
    column: int
    
    def __str__(self) -> str:
        return f"{self.line}:{self.column}"

@dataclass
class Range:
    """Range in source code"""
    start: Position
    end: Position
    
    def __str__(self) -> str:
        return f"{self.start}-{self.end}"

@dataclass
class UnifiedNode:
    """Node in the unified AST"""
    type: NodeType
    name: str = ""
    value: Optional[Any] = None
    source_range: Optional[Range] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List['UnifiedNode'] = field(default_factory=list)
    parent: Optional['UnifiedNode'] = None
    language: str = ""
    
    def add_child(self, child: 'UnifiedNode') -> 'UnifiedNode':
        """
        Add a child node
        
        Args:
            child: Child node
            
        Returns:
            The child node
        """
        child.parent = self
        self.children.append(child)
        return child
    
    def find_by_type(self, node_type: NodeType) -> List['UnifiedNode']:
        """
        Find all nodes of a specific type
        
        Args:
            node_type: Node type to find
            
        Returns:
            List of matching nodes
        """
        result = []
        if self.type == node_type:
            result.append(self)
        
        for child in self.children:
            result.extend(child.find_by_type(node_type))
        
        return result
    
    def find_by_name(self, name: str) -> List['UnifiedNode']:
        """
        Find all nodes with a specific name
        
        Args:
            name: Node name to find
            
        Returns:
            List of matching nodes
        """
        result = []
        if self.name == name:
            result.append(self)
        
        for child in self.children:
            result.extend(child.find_by_name(name))
        
        return result
    
    def find_by_path(self, path: List[str]) -> Optional['UnifiedNode']:
        """
        Find a node by path
        
        Args:
            path: Path components
            
        Returns:
            Matching node or None
        """
        if not path:
            return self
        
        current_path = path[0]
        remaining_path = path[1:]
        
        for child in self.children:
            if child.name == current_path:
                return child.find_by_path(remaining_path)
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dictionary representation
        """
        result = {
            "type": self.type.name,
            "name": self.name
        }
        
        if self.value is not None:
            result["value"] = self.value
            
        if self.source_range:
            result["source_range"] = {
                "start": {"line": self.source_range.start.line, "column": self.source_range.start.column},
                "end": {"line": self.source_range.end.line, "column": self.source_range.end.column}
            }
            
        if self.attributes:
            result["attributes"] = self.attributes
            
        if self.language:
            result["language"] = self.language
            
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional['UnifiedNode'] = None) -> 'UnifiedNode':
        """
        Create from dictionary
        
        Args:
            data: Dictionary representation
            parent: Parent node
            
        Returns:
            Created node
        """
        source_range = None
        if "source_range" in data:
            source_range = Range(
                start=Position(
                    line=data["source_range"]["start"]["line"],
                    column=data["source_range"]["start"]["column"]
                ),
                end=Position(
                    line=data["source_range"]["end"]["line"],
                    column=data["source_range"]["end"]["column"]
                )
            )
            
        node = cls(
            type=NodeType[data["type"]],
            name=data.get("name", ""),
            value=data.get("value"),
            source_range=source_range,
            attributes=data.get("attributes", {}),
            language=data.get("language", ""),
            parent=parent
        )
        
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, parent=node)
            node.children.append(child)
            
        return node
    
    def to_json(self) -> str:
        """
        Convert to JSON
        
        Returns:
            JSON representation
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UnifiedNode':
        """
        Create from JSON
        
        Args:
            json_str: JSON representation
            
        Returns:
            Created node
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

class UnifiedAST:
    """Unified Abstract Syntax Tree"""
    
    def __init__(self, root: Optional[UnifiedNode] = None):
        """
        Initialize the AST
        
        Args:
            root: Root node
        """
        self.root = root or UnifiedNode(type=NodeType.ROOT, name="root")
    
    def find_by_type(self, node_type: NodeType) -> List[UnifiedNode]:
        """
        Find all nodes of a specific type
        
        Args:
            node_type: Node type to find
            
        Returns:
            List of matching nodes
        """
        return self.root.find_by_type(node_type)
    
    def find_by_name(self, name: str) -> List[UnifiedNode]:
        """
        Find all nodes with a specific name
        
        Args:
            name: Node name to find
            
        Returns:
            List of matching nodes
        """
        return self.root.find_by_name(name)
    
    def find_by_path(self, path: List[str]) -> Optional[UnifiedNode]:
        """
        Find a node by path
        
        Args:
            path: Path components
            
        Returns:
            Matching node or None
        """
        return self.root.find_by_path(path)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dictionary representation
        """
        return self.root.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedAST':
        """
        Create from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created AST
        """
        root = UnifiedNode.from_dict(data)
        return cls(root)
    
    def to_json(self) -> str:
        """
        Convert to JSON
        
        Returns:
            JSON representation
        """
        return self.root.to_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UnifiedAST':
        """
        Create from JSON
        
        Args:
            json_str: JSON representation
            
        Returns:
            Created AST
        """
        root = UnifiedNode.from_json(json_str)
        return cls(root)
    
    def merge(self, other: 'UnifiedAST') -> 'UnifiedAST':
        """
        Merge with another AST
        
        Args:
            other: Other AST
            
        Returns:
            Merged AST
        """
        # Create a new AST
        merged = UnifiedAST()
        
        # Clone this AST's nodes
        for child in self.root.children:
            merged.root.add_child(_clone_node(child))
        
        # Add other AST's nodes
        for child in other.root.children:
            merged.root.add_child(_clone_node(child))
        
        return merged

def _clone_node(node: UnifiedNode) -> UnifiedNode:
    """
    Clone a node and its children
    
    Args:
        node: Node to clone
        
    Returns:
        Cloned node
    """
    clone = UnifiedNode(
        type=node.type,
        name=node.name,
        value=node.value,
        source_range=node.source_range,
        attributes=node.attributes.copy(),
        language=node.language
    )
    
    for child in node.children:
        clone.add_child(_clone_node(child))
    
    return clone

###############################
# Language Adapters
###############################

class LanguageAdapter(ABC):
    """Base class for language adapters"""
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Language identifier"""
        pass
    
    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """File extensions handled by this adapter"""
        pass
    
    @abstractmethod
    async def parse(self, code: str, file