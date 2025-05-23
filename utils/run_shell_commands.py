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
import traceback

def get_system_python_paths():
    """
    Returns a list of directories containing Python packages in the system
    excluding the virtual environment.
    
    The function works by iterating over common base directories for Python
    packages and checking if they exist. The directories are then added to a
    list which is returned.
    """
    
    # Get Python version
    python_version = subprocess.run(
        ["/usr/bin/python3", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    # Common base directories for Python packages
    base_dirs = [
        "/usr/lib/python3",  # Default installation directory for Python packages
        "/usr/local/lib/python3",  # Directory for user-installed packages
        f"/usr/lib/python{python_version}",  # Specific Python version directory
        f"/usr/local/lib/python{python_version}"  # Specific Python version directory
    ]
    
    # Common package directories
    package_dirs = [
        "dist-packages",  # Debian/Ubuntu packages
        "site-packages"  # Python packages installed using pip
    ]
    
    # Find all existing paths
    system_paths = []
    for base in base_dirs:
        for package_dir in package_dirs:
            path = os.path.join(base, package_dir)
            if os.path.exists(path):
                system_paths.append(path)
                
        # Also check for direct dist-packages in python3 directory
        if os.path.exists(base) and os.path.isdir(base):
            system_paths.append(base)
            
    return list(set(system_paths))  # Remove duplicates

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

def run_shell_command_without_virtualenv(command):
    """
    Runs a shell command without using a virtual environment.

    This function is useful when a shell command needs to be run without
    using the dependencies installed in the virtual environment. It
    temporarily modifies the environment to include the system Python
    paths and then runs the command with the modified environment.

    Args:
        command (str): The shell command to run.

    Returns:
        subprocess.CompletedProcess: The result of the command execution.
    """
    # Get the original PYTHONPATH
    original_pythonpath = os.environ.get('PYTHONPATH', '')
    
    try:
        # Temporarily modify the environment
        modified_env = os.environ.copy()
        
        # Get system Python paths automatically
        system_paths = get_system_python_paths()
        
        if not system_paths:
            raise Exception("No system Python paths found")
            
        modified_env['PYTHONPATH'] = ':'.join(system_paths + [original_pythonpath])
        
        # Run the command with modified environment
        cmd = f"/usr/bin/python3 {command}"
        result = subprocess.run(
            cmd,
            shell=True,
            env=modified_env,
            capture_output=True,
            text=True
        )

        return result
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def run_shell_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        if(result.stderr):
            raise Exception(result.stderr)

        return result
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("Error in run_shell_command, running  run_shell_command_without_virtualenv")
        return run_shell_command_without_virtualenv(command)