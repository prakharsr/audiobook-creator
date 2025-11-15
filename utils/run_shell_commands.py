"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import subprocess
import shutil
import os
import sys
import traceback
import shlex
import re

def check_if_calibre_is_installed():
    """
    Checks if Calibre is installed.
    
    Returns True if Calibre is installed and False otherwise.
    """
    # Check if Calibre is installed by checking if either the `calibre` or
    # `ebook-convert` command is available in the PATH.
    calibre_installed = shutil.which("calibre") or shutil.which("ebook-convert")
    
    if calibre_installed:
        return True
    else:
        return False
    
def check_if_ffmpeg_is_installed():
    """
    Checks if FFmpeg is installed.

    Returns True if FFmpeg is installed and False otherwise.
    """
    ffmpeg_installed = shutil.which("ffmpeg")
    
    if ffmpeg_installed:
        # If the command is available in the PATH, FFmpeg is installed
        return True
    else:
        # If the command is not available in the PATH, FFmpeg is not installed
        return False

def validate_file_path_allowlist(file_path):
    """
    Validates file paths using allowlist regex patterns (secure approach).
    Only allows safe file path patterns and rejects everything else.
    
    Args:
        file_path (str): The file path to validate
        
    Returns:
        bool: True if path matches safe patterns, False otherwise
    """
    if not file_path or not isinstance(file_path, str):
        return False
    
    # Allowlist pattern for safe file paths
    # Allows: letters, numbers, spaces, hyphens, underscores, dots, forward slashes, commas
    # Specifically excludes shell metacharacters and command injection patterns
    safe_path_pattern = r"^[a-zA-Z0-9\s\-_.:/'\\,]+\.[a-zA-Z0-9]{1,10}$|^[a-zA-Z0-9\s\-_.:/'\\,]+/$"
    
    # Additional check for relative path traversal
    safe_relative_pattern = r"^(?!.*\.\.)[a-zA-Z0-9\s\-_.:/'\\,]+$"
    
    return (re.match(safe_path_pattern, file_path) is not None and 
            re.match(safe_relative_pattern, file_path) is not None)

def validate_command_arguments_allowlist(args):
    """
    Validates command arguments using allowlist patterns.
    
    Args:
        args (list): List of command arguments
        
    Returns:
        bool: True if all arguments are safe, False otherwise
    """
    if not isinstance(args, list):
        return False
    
    for arg in args:
        if not isinstance(arg, str):
            return False

        # Disallow directory traversal early
        if ".." in arg:
            return False
            
        # Allow safe argument patterns:
        safe_arg_patterns = [
            # File paths and extensions (no '..')
            r"^(?!.*\.\.)[a-zA-Z0-9\s\-_.:/'\\,]+\.[a-zA-Z0-9]{1,10}$",
            # Directory paths (no '..')
            r"^(?!.*\.\.)[a-zA-Z0-9\s\-_.:/'\\,]+/?$",
            # Command flags like -y, --verbose, -map_metadata
            r'^-{1,2}[a-zA-Z0-9\-_:]+$',
            # Numbers with optional size suffixes and standalone numbers
            r'^\d+[kmgtKMGT]?$',
            # Boolean values
            r'^(true|false|yes|no|on|off)$',
            # FFmpeg-style arguments with colons (like disposition:v:0)
            r'^[a-zA-Z0-9\-_]+:[a-zA-Z0-9\-_:]+$',
            # Basic alphanumeric values with safe punctuation
            r'^[a-zA-Z0-9\-_+=:,]+$',
            # Metadata values (key=value with spaces, apostrophes, brackets)
            r'^[a-zA-Z0-9\-_]+=[\w\s\-_.,:()[\]\'\"!?]+$',
            # Date/time formats (ISO 8601 style)
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$',
            # Language codes and simple identifiers
            r'^[a-zA-Z]{2,3}$',
            # Simple words and phrases with common punctuation
            r'^[\w\s\-_.()[\]\'\"!?,:]+$',
        ]
        
        # Check if argument matches any safe pattern
        is_safe = any(re.match(pattern, arg, re.IGNORECASE) for pattern in safe_arg_patterns)
        
        if not is_safe:
            # For debugging: print which argument failed validation
            # print(f"Argument failed validation: {arg}")
            return False
    
    return True

def validate_command_safety(command):
    """
    Validates that a command is safe using allowlist approach.
    
    Args:
        command (str or list): The command to validate
        
    Returns:
        bool: True if command appears safe, False otherwise
    """
    if isinstance(command, str):
        try:
            cmd_parts = shlex.split(command)
        except ValueError:
            return False  # Invalid shell syntax
    else:
        cmd_parts = command
    
    if not cmd_parts or len(cmd_parts) == 0:
        return False
    
    # Validate command name (first argument)
    command_name = cmd_parts[0]
    
    # Allow common safe installation paths and simple command names
    safe_command_patterns = [
        r'^[a-zA-Z0-9\-_]+$',  # Simple command names like 'ffmpeg', 'which'
        r'^[a-zA-Z0-9\-_./]+$',  # Paths with basic separators
        r'^/usr/bin/[a-zA-Z0-9\-_]+$',  # Standard /usr/bin/ paths
        r'^/usr/local/bin/[a-zA-Z0-9\-_]+$',  # Local installation paths
        r'^/opt/[a-zA-Z0-9\-_.]+/[a-zA-Z0-9\-_./]+$',  # /opt/ installations
        r'^/Applications/[a-zA-Z0-9\-_.]+\.app/Contents/MacOS/[a-zA-Z0-9\-_]+$',  # macOS app bundles
        r'^C:\\Program Files\\[a-zA-Z0-9\-_.]+\\[a-zA-Z0-9\-_]+\.exe$',  # windows installations
    ]
    
    # Check if command name matches any safe pattern
    is_safe_command = any(re.match(pattern, command_name, re.IGNORECASE) for pattern in safe_command_patterns)
    
    if not is_safe_command:
        return False
    
    # Validate all arguments using allowlist
    if len(cmd_parts) > 1:
        return validate_command_arguments_allowlist(cmd_parts[1:])
    
    return True

def run_shell_command_secure(command, allowed_commands=None):
    """
    Securely runs a shell command using list-based subprocess calls.
    
    Args:
        command (str or list): The command to run
        allowed_commands (list): List of allowed command names (optional)
        
    Returns:
        subprocess.CompletedProcess: The result of the command execution
    """
    try:
        # Convert string command to list for security
        if isinstance(command, str):
            # For simple commands, try to parse safely
            cmd_parts = shlex.split(command)
        else:
            cmd_parts = command
            
        if not cmd_parts or len(cmd_parts) == 0:
            return False
            
        # Validate command safety
        if not validate_command_safety(cmd_parts):
            raise ValueError(f"Command contains potentially dangerous patterns: {cmd_parts}")
            
        # Check if command is in allowed list (compare basename for full paths)
        if allowed_commands:
            command_name = cmd_parts[0]
            # Extract basename from full path for comparison
            command_basename = os.path.basename(command_name)
            # if on windows strip off .exe to compare
            if sys.platform == 'win32' and command_basename.lower().endswith(".exe"):
                command_basename = command_basename[:-4]
            
            if command_basename not in allowed_commands:
                raise ValueError(f"Command '{command_basename}' not in allowed commands list")
            
        # Execute with secure subprocess call (no shell=True)
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            env=os.environ.copy() # Pass environment variables explicitly
        )
        
        if result.stderr and result.returncode != 0:
            raise Exception(result.stderr)

        return result
        
    except Exception as e:
        print(f"Error in run_shell_command_secure: {e}")
        traceback.print_exc()
        return None