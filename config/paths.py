#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Path configuration module for managing data and output directories.

Author: Michael Chen
Date: 2024-01-10
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".recontext", "cache")
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".recontext", "data")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), ".recontext", "output")
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".recontext", "models")


def get_cache_dir(custom_dir: Optional[str] = None) -> str:
    """Get cache directory path.
    
    Args:
        custom_dir: Optional custom directory path
        
    Returns:
        Absolute path to cache directory
    """
    cache_dir = custom_dir or os.environ.get("RECONTEXT_CACHE_DIR", DEFAULT_CACHE_DIR)
    
    # Create directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.abspath(cache_dir)


def get_data_dir(custom_dir: Optional[str] = None) -> str:
    """Get data directory path.
    
    Args:
        custom_dir: Optional custom directory path
        
    Returns:
        Absolute path to data directory
    """
    data_dir = custom_dir or os.environ.get("RECONTEXT_DATA_DIR", DEFAULT_DATA_DIR)
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    return os.path.abspath(data_dir)


def get_output_dir(custom_dir: Optional[str] = None) -> str:
    """Get output directory path.
    
    Args:
        custom_dir: Optional custom directory path
        
    Returns:
        Absolute path to output directory
    """
    output_dir = custom_dir or os.environ.get("RECONTEXT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.abspath(output_dir)


def get_model_dir(custom_dir: Optional[str] = None) -> str:
    """Get model directory path.
    
    Args:
        custom_dir: Optional custom directory path
        
    Returns:
        Absolute path to model directory
    """
    model_dir = custom_dir or os.environ.get("RECONTEXT_MODEL_DIR", DEFAULT_MODEL_DIR)
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    return os.path.abspath(model_dir)


def get_project_root() -> str:
    """Get project root directory path.
    
    Returns:
        Absolute path to project root directory
    """
    # Get path to current file
    current_file = Path(__file__).resolve()
    
    # Get parent directory (config/)
    config_dir = current_file.parent
    
    # Get parent directory (project root)
    project_root = config_dir.parent
    
    return str(project_root)


def get_package_path(package_name: str) -> str:
    """Get package directory path.
    
    Args:
        package_name: Package name
        
    Returns:
        Absolute path to package directory
    """
    # Get project root
    project_root = get_project_root()
    
    # Get package directory
    package_dir = os.path.join(project_root, package_name)
    
    if not os.path.isdir(package_dir):
        logger.warning(f"Package directory not found: {package_dir}")
    
    return package_dir


def get_package_dirs() -> Dict[str, str]:
    """Get paths to all package directories.
    
    Returns:
        Dictionary of package names and paths
    """
    # Get project root
    project_root = get_project_root()
    
    # Get directories
    package_dirs = {}
    
    # Main packages
    main_packages = ["core", "semantics", "integration", "language", "visualization", "utils", "config"]
    
    for package in main_packages:
        package_dir = os.path.join(project_root, "recontext", package)
        if os.path.isdir(package_dir):
            package_dirs[package] = package_dir
    
    return package_dirs


def get_config_path(config_name: str) -> str:
    """Get path to configuration file.
    
    Args:
        config_name: Configuration filename
        
    Returns:
        Absolute path to configuration file
    """
    # Get project root
    project_root = get_project_root()
    
    # Get config directory
    config_dir = os.path.join(project_root, "recontext", "config")
    
    # Get config file path
    config_path = os.path.join(config_dir, config_name)
    
    if not config_name.endswith((".yaml", ".yml", ".json")):
        # Try different extensions
        for ext in [".yaml", ".yml", ".json"]:
            test_path = config_path + ext
            if os.path.isfile(test_path):
                config_path = test_path
                break
    
    if not os.path.isfile(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
    
    return config_path


def get_default_config_path() -> str:
    """Get path to default configuration file.
    
    Returns:
        Absolute path to default configuration file
    """
    return get_config_path("default_config.yaml")


def get_sample_data_path(sample_name: str) -> str:
    """Get path to sample data.
    
    Args:
        sample_name: Sample data name
        
    Returns:
        Absolute path to sample data
    """
    # Get data directory
    data_dir = get_data_dir()
    
    # Get samples directory
    samples_dir = os.path.join(data_dir, "samples")
    
    # Create directory if it doesn't exist
    os.makedirs(samples_dir, exist_ok=True)
    
    # Get sample data path
    sample_path = os.path.join(samples_dir, sample_name)
    
    return sample_path


def get_log_path(log_name: str = "recontext.log") -> str:
    """Get path to log file.
    
    Args:
        log_name: Log filename
        
    Returns:
        Absolute path to log file
    """
    # Get output directory
    output_dir = get_output_dir()
    
    # Get logs directory
    logs_dir = os.path.join(output_dir, "logs")
    
    # Create directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get log file path
    log_path = os.path.join(logs_dir, log_name)
    
    return log_path


def get_temp_dir() -> str:
    """Get temporary directory path.
    
    Returns:
        Absolute path to temporary directory
    """
    import tempfile
    
    # Get base temp directory
    base_temp_dir = tempfile.gettempdir()
    
    # Create recontext-specific temp directory
    temp_dir = os.path.join(base_temp_dir, "recontext")
    
    # Create directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    return temp_dir


def print_paths() -> None:
    """Print all path configurations."""
    # Get paths
    project_root = get_project_root()
    cache_dir = get_cache_dir()
    data_dir = get_data_dir()
    output_dir = get_output_dir()
    model_dir = get_model_dir()
    default_config_path = get_default_config_path()
    temp_dir = get_temp_dir()
    log_path = get_log_path()
    
    # Print paths
    print("RECONTEXT Path Configuration:")
    print(f"  Project Root: {project_root}")
    print(f"  Cache Directory: {cache_dir}")
    print(f"  Data Directory: {data_dir}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Model Directory: {model_dir}")
    print(f"  Default Config: {default_config_path}")
    print(f"  Temp Directory: {temp_dir}")
    print(f"  Log File: {log_path}")


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="RECONTEXT path configuration")
    parser.add_argument("--print", action="store_true", help="Print all path configurations")
    parser.add_argument("--get", help="Get specific path (cache, data, output, model, config, root)")
    
    args = parser.parse_args()
    
    if args.print:
        print_paths()
    elif args.get:
        if args.get == "cache":
            print(get_cache_dir())
        elif args.get == "data":
            print(get_data_dir())
        elif args.get == "output":
            print(get_output_dir())
        elif args.get == "model":
            print(get_model_dir())
        elif args.get == "config":
            print(get_default_config_path())
        elif args.get == "root":
            print(get_project_root())
        else:
            logger.error(f"Unknown path type: {args.get}")
    else:
        print_paths()