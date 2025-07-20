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
import re
import os
import traceback
import shlex
import json
from utils.run_shell_commands import run_shell_command_secure, validate_file_path_allowlist

# Escape double quotes by replacing them with \"
def escape_metadata(value):
    if value:
        # Remove any potentially dangerous characters for metadata
        # Keep only alphanumeric, spaces, common punctuation
        safe_chars = re.sub(r'[^\w\s\-.,!?\'"()[\]:;]', '', str(value))
        return safe_chars.replace('"', '\\"')  # Escape double quotes
    return ""

def validate_file_path(file_path):
    """
    Validates that a file path is safe using allowlist approach.
    
    Args:
        file_path (str): The file path to validate
        
    Returns:
        bool: True if path is safe, False otherwise
    """
    if not file_path or not isinstance(file_path, str):
        return False
    
    # Use allowlist-based validation
    if not validate_file_path_allowlist(file_path):
        return False
        
    # Check if file exists and is readable
    try:
        return os.path.exists(file_path) and os.access(file_path, os.R_OK)
    except (OSError, TypeError):
        return False

def get_ebook_metadata_with_cover(book_path):
    """
    Extracts metadata from an ebook and saves its cover image.

    Args:
        book_path (str): The path to the ebook file.

    Returns:
        dict: A dictionary containing the ebook's metadata.
    """
    # Validate file path
    if not validate_file_path(book_path):
        raise ValueError(f"Invalid or unsafe book path: {book_path}")
        
    # Get ebook-meta binary path securely
    allowed_commands = ['which', 'ebook-meta']
    ebook_meta_bin_result = run_shell_command_secure("which ebook-meta", allowed_commands)
    
    if not ebook_meta_bin_result or not ebook_meta_bin_result.stdout.strip():
        raise RuntimeError("ebook-meta command not found")
        
    ebook_meta_bin_path = ebook_meta_bin_result.stdout.strip()

    # Build secure command as list
    command = [ebook_meta_bin_path, book_path, "--get-cover", "cover.jpg"]

    # Run the command securely using our centralized function
    allowed_ebook_commands = ['ebook-meta']
    result = run_shell_command_secure(command, allowed_ebook_commands)
    
    if not result:
        raise RuntimeError("Failed to extract metadata")

    metadata = {}
    # Parse the command output to extract metadata
    for line in result.stdout.split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            metadata[key.strip()] = value.strip()
    
    return metadata
    
def get_audio_duration_using_ffprobe(file_path):
    """
    Returns the duration of an audio file in milliseconds using ffprobe.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        int: The duration of the audio file in milliseconds.
    """
    # Validate file path
    if not validate_file_path(file_path):
        raise ValueError(f"Invalid or unsafe file path: {file_path}")
        
    # Construct the command to execute
    cmd = [
        "ffprobe",  # Use ffprobe to get the duration
        "-v", "error",  # Set the verbosity to error
        "-show_entries",  # Show the specified entries
        "format=duration",  # Show the duration
        "-of",  # Specify the output format
        "default=noprint_wrappers=1:nokey=1",  # Print the duration without any additional information
        file_path  # Specify the file to analyze
    ]
    
    # Run the command securely
    allowed_commands = ['ffprobe']
    result = run_shell_command_secure(cmd, allowed_commands)
    
    if not result or result.returncode != 0:
        raise RuntimeError(f"Failed to get audio duration: {result.stderr if result else 'Unknown error'}")
        
    # Convert the output to an integer (in milliseconds) and return it
    return int(float(result.stdout.strip()) * 1000)

def get_audio_duration_using_raw_ffmpeg(file_path):
    """Returns the duration of an audio file using FFmpeg."""
    # Validate file path
    if not validate_file_path(file_path):
        raise ValueError(f"Invalid or unsafe file path: {file_path}")
        
    cmd = ["ffmpeg", "-y", "-i", file_path, "-f", "null", "-"]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result:
            return None
            
        stderr_output = result.stderr

        # Look for the final timestamp (time=xx:xx:xx.xx) in FFmpeg output
        match = re.search(r"time=(\d+):(\d+):([\d.]+)", stderr_output)
        if match:
            hours, minutes, seconds = map(float, match.groups())
            return int(float((hours * 3600 + minutes * 60 + seconds) * 1000))
        else:
            return None  # Duration not found

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None

def generate_chapters_file(chapter_files, output_file="chapters.txt"):
    """
    Generates a chapter metadata file for FFmpeg.

    The chapter metadata file is a text file that contains information about each chapter in the audiobook, such as the chapter title and the start and end times of the chapter.

    Args:
        chapter_files (list): A list of the paths to the individual chapter audio files.
        output_file (str): The path to the output chapter metadata file. Defaults to "chapters.txt".
    """
    start_time = 0
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for chapter in chapter_files:
            duration = get_audio_duration_using_ffprobe(os.path.join("temp_audio", chapter))
            end_time = start_time + duration
            
            # Write the chapter metadata to the file
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_time}\n")
            f.write(f"END={end_time}\n")
            f.write(f"title={os.path.splitext(chapter)[0]}\n\n")  # Use filename as chapter title
            
            # Update the start time for the next chapter
            start_time = end_time

def create_m4a_file_from_raw_aac_file(input_file_path, output_file_path):
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    cmd = ["ffmpeg", "-y", "-i", input_file_path, "-c", "copy", output_file_path]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating M4A from AAC: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def create_m4a_file_from_wav_file(input_file_path, output_file_path):
    """
    Convert WAV to M4A using AAC encoding with intelligent quality settings.
    Preserves sample rate and channel layout from input file.
    """
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    # Get properties of input file to preserve them
    audio_props = get_audio_properties(input_file_path)
    sample_rate = audio_props["sample_rate"]
    
    cmd = [
        "ffmpeg", "-y", "-i", input_file_path, 
        "-c:a", "aac", "-b:a", "256k",  # High quality AAC
        "-ar", str(sample_rate),  # Preserve original sample rate
        output_file_path
    ]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating M4A from WAV: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None

def create_aac_file_from_m4a_file(input_file_path, output_file_path):
    """
    Extract AAC stream from M4A container (lossless container change only).
    """
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    cmd = ["ffmpeg", "-y", "-i", input_file_path, "-c", "copy", output_file_path]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating AAC from M4A: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def create_mp3_file_from_m4a_file(input_file_path, output_file_path):
    """
    Convert M4A to MP3 preserving original sample rate and quality.
    Uses high-quality LAME encoder settings.
    """
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    # Get properties of input file
    audio_props = get_audio_properties(input_file_path)
    sample_rate = audio_props["sample_rate"]
    
    # Choose appropriate bitrate based on sample rate and content type (speech)
    bitrate = "192k" if sample_rate >= 44100 else "160k"
    
    cmd = [
        "ffmpeg", "-y", "-i", input_file_path, 
        "-c:a", "libmp3lame", "-b:a", bitrate,
        "-ar", str(sample_rate),  # Preserve original sample rate
        "-q:a", "2",  # High quality VBR setting
        output_file_path
    ]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating MP3 from M4A: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def create_wav_file_from_m4a_file(input_file_path, output_file_path):
    """
    Convert M4A to WAV preserving original sample rate and channel layout.
    This is a lossless conversion that maintains audio quality.
    """
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    # Get properties of input file to preserve them
    audio_props = get_audio_properties(input_file_path)
    sample_rate = audio_props["sample_rate"]
    
    cmd = [
        "ffmpeg", "-y", "-i", input_file_path, 
        "-c:a", "pcm_s16le",  # 16-bit PCM (standard WAV)
        "-ar", str(sample_rate),  # Preserve original sample rate
        # Note: Not specifying channel count to preserve original layout
        output_file_path
    ]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating WAV from M4A: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def create_opus_file_from_m4a_file(input_file_path, output_file_path):
    """
    Convert M4A to Opus with intelligent quality settings.
    Opus only supports specific sample rates, so we map to the closest supported rate.
    """
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    # Get properties of input file
    audio_props = get_audio_properties(input_file_path)
    original_sample_rate = audio_props["sample_rate"]
    
    # Opus supported sample rates: 8000, 12000, 16000, 24000, 48000
    # Map input sample rates to closest supported Opus rate
    if original_sample_rate <= 10000:
        opus_sample_rate = 8000
        bitrate = "64k"
    elif original_sample_rate <= 14000:
        opus_sample_rate = 12000
        bitrate = "80k"
    elif original_sample_rate <= 20000:
        opus_sample_rate = 16000
        bitrate = "96k"
    elif original_sample_rate <= 36000:
        opus_sample_rate = 24000
        bitrate = "128k"
    else:
        opus_sample_rate = 48000  # For 44100Hz and higher
        bitrate = "160k"
    
    cmd = [
        "ffmpeg", "-y", "-i", input_file_path, 
        "-c:a", "libopus", "-b:a", bitrate,
        "-ar", str(opus_sample_rate),  # Use compatible Opus sample rate
        output_file_path
    ]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating Opus from M4A: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def create_flac_file_from_m4a_file(input_file_path, output_file_path):
    """
    Convert M4A to FLAC (lossless compression).
    Preserves all original audio properties for maximum quality.
    """
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    # Get properties of input file to preserve them
    audio_props = get_audio_properties(input_file_path)
    sample_rate = audio_props["sample_rate"]
    
    cmd = [
        "ffmpeg", "-y", "-i", input_file_path, 
        "-c:a", "flac",
        "-ar", str(sample_rate),  # Preserve original sample rate
        # FLAC is lossless so we preserve everything
        output_file_path
    ]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating FLAC from M4A: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def create_pcm_file_from_m4a_file(input_file_path, output_file_path):
    """
    Convert M4A to raw PCM preserving original audio properties.
    This creates uncompressed audio data with maximum quality.
    """
    # Validate file paths
    if not validate_file_path(input_file_path):
        raise ValueError(f"Invalid input file path: {input_file_path}")
        
    # Get properties of input file to preserve them
    audio_props = get_audio_properties(input_file_path)
    sample_rate = audio_props["sample_rate"]
    channels = audio_props["channels"]
    
    cmd = [
        "ffmpeg", "-y", "-i", input_file_path, 
        "-f", "s16le",  # 16-bit little-endian format
        "-acodec", "pcm_s16le", 
        "-ar", str(sample_rate),  # Preserve original sample rate
        "-ac", str(channels),  # Preserve original channel count
        output_file_path
    ]
    
    try:
        # Run the command securely
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            print(f"Error creating PCM from M4A: {error_msg}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    
def convert_audio_file_formats(input_format, output_format, folder_path, file_name):
        input_path = os.path.join(folder_path, f"{file_name}.{input_format}")
        output_path = os.path.join(folder_path, f"{file_name}.{output_format}")

        if output_format == "aac":
            create_aac_file_from_m4a_file(input_path, output_path)
        elif output_format == "m4a":
            if input_format == "aac":
                create_m4a_file_from_raw_aac_file(input_path, output_path)
            elif input_format == "m4a":
                pass # Already generated
            elif input_format == "wav":
                create_m4a_file_from_wav_file(input_path, output_path)
        elif output_format == "mp3":
            create_mp3_file_from_m4a_file(input_path, output_path)
        elif output_format == "wav":
            create_wav_file_from_m4a_file(input_path, output_path)
        elif output_format == "opus":
            create_opus_file_from_m4a_file(input_path, output_path)
        elif output_format == "flac":
            create_flac_file_from_m4a_file(input_path, output_path)
        elif output_format == "pcm":
            create_pcm_file_from_m4a_file(input_path, output_path)
    
def merge_chapters_to_m4b(book_path, chapter_files):
    """
    Uses ffmpeg to merge all chapter files into an M4B audiobook.

    This function takes the path to the book file and a list of chapter files as input, and generates an M4B audiobook with chapter metadata and a cover image.

    Args:
        book_path (str): The path to the book file.
        chapter_files (list): A list of the paths to the individual chapter audio files.
    """
    # Validate inputs
    if not validate_file_path(book_path):
        raise ValueError(f"Invalid or unsafe book path: {book_path}")
        
    file_list_path = "chapter_list.txt"
    
    with open(file_list_path, "w", encoding='utf-8') as f:
        for chapter in chapter_files:
            # Validate each chapter file path
            chapter_path = os.path.join('temp_audio', chapter)
            if not validate_file_path(chapter_path):
                raise ValueError(f"Invalid chapter file: {chapter}")
            f.write(f"file '{chapter_path}'\n")

    metadata = get_ebook_metadata_with_cover(book_path)
    title = escape_metadata(metadata.get("Title", ""))
    authors = escape_metadata(metadata.get("Author(s)", ""))
    publisher = escape_metadata(metadata.get("Publisher", ""))
    languages = escape_metadata(metadata.get("Languages", ""))
    published_date = escape_metadata(metadata.get("Published", ""))
    comments = escape_metadata(metadata.get("Comments", ""))
    
    # Generate chapter metadata
    generate_chapters_file(chapter_files, "chapters.txt")

    output_m4b = "generated_audiobooks/audiobook.m4b"
    cover_image = "cover.jpg"

    # Validate required files exist
    for required_file in [file_list_path, cover_image, "chapters.txt"]:
        if not validate_file_path(required_file):
            raise ValueError(f"Required file missing or invalid: {required_file}")

    # Build secure FFmpeg command as list
    ffmpeg_cmd = [
        "ffmpeg", "-y", 
        "-f", "concat", "-safe", "0", "-i", file_list_path,
        "-i", cover_image, 
        "-i", "chapters.txt",
        "-c", "copy", 
        "-map", "0", 
        "-map", "1", 
        "-disposition:v:0", "attached_pic", 
        "-map_metadata", "2"
    ]
    
    # Add metadata safely
    if title:
        ffmpeg_cmd.extend(["-metadata", f"title={title}"])
    if authors:
        ffmpeg_cmd.extend(["-metadata", f"artist={authors}"])
    if title:
        ffmpeg_cmd.extend(["-metadata", f"album={title}"])
    ffmpeg_cmd.extend(["-metadata", "genre=Audiobook"])
    if publisher:
        ffmpeg_cmd.extend(["-metadata", f"publisher={publisher}"])
    if languages:
        ffmpeg_cmd.extend(["-metadata", f"language={languages}"])
    if published_date:
        ffmpeg_cmd.extend(["-metadata", f"date={published_date}"])
    if comments:
        ffmpeg_cmd.extend(["-metadata", f"description={comments}"])
        
    ffmpeg_cmd.append(output_m4b)
    
    # Use centralized secure command execution
    allowed_ffmpeg_commands = ['ffmpeg']
    result = run_shell_command_secure(ffmpeg_cmd, allowed_ffmpeg_commands)
    
    if not result or result.returncode != 0:
        error_msg = result.stderr if result else "Unknown error"
        raise RuntimeError(f"FFmpeg failed: {error_msg}")
        
    print(f"Audiobook created: {output_m4b}")

def add_silence_to_audio_file_by_appending_pre_generated_silence(temp_dir, input_file_name):
    silence_path = "static_files/silence.aac" # Pre generated 1 seconds of silence using command `ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -c:a aac silence.aac`

    with open(silence_path, "rb") as silence_file, open(f"{temp_dir}/{input_file_name}", "ab") as audio_file:
        audio_file.write(silence_file.read())  # Append silence to the end of the audio file

def add_silence_to_audio_file_by_reencoding_using_ffmpeg(temp_dir, input_file_name, pause_duration):
    """
    Adds a silence of specified duration at the end of an audio file.

    Args:
        temp_dir (str): The temporary directory to store the silence file.
        input_file_name (str): The name of the file to add silence to.
        pause_duration (str): The duration of the silence (e.g. 00:00:05).
    """
    # Validate inputs
    if not validate_file_path(temp_dir) or not input_file_name:
        raise ValueError("Invalid temporary directory or file name")
        
    # Validate pause_duration format (basic check)
    if not re.match(r'^\d{2}:\d{2}:\d{2}$', pause_duration):
        raise ValueError("Invalid pause duration format. Expected HH:MM:SS")
    
    silence_path = os.path.join(temp_dir, "silence.aac")
    input_path = os.path.join(temp_dir, input_file_name)
    
    # Validate file paths
    if not validate_file_path(input_path):
        raise ValueError(f"Invalid input file path: {input_path}")
    
    # Generate a silence file with the specified duration - secure command
    generate_silence_command = [
        "ffmpeg", "-y", 
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", 
        "-t", pause_duration, 
        "-c:a", "aac", 
        silence_path
    ]
    
    # Use centralized secure command execution
    allowed_ffmpeg_commands = ['ffmpeg']
    result = run_shell_command_secure(generate_silence_command, allowed_ffmpeg_commands)
    
    if not result or result.returncode != 0:
        error_msg = result.stderr if result else "Unknown error"
        raise RuntimeError(f"Failed to generate silence: {error_msg}")

    # Add the silence to the end of the audio file - secure command
    temp_output_path = os.path.join(temp_dir, "temp_audio_file.aac")
    add_silence_command = [
        "ffmpeg", "-y", 
        "-i", input_path, 
        "-i", silence_path, 
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]", 
        "-map", "[out]", 
        temp_output_path
    ]
    
    # Use centralized secure command execution
    result = run_shell_command_secure(add_silence_command, allowed_ffmpeg_commands)
    
    if not result or result.returncode != 0:
        error_msg = result.stderr if result else "Unknown error"
        raise RuntimeError(f"Failed to add silence: {error_msg}")

    # Rename the temporary file back to the original file name
    try:
        os.replace(temp_output_path, input_path)
    except OSError as e:
        raise RuntimeError(f"Failed to rename file: {e}")

def merge_chapters_to_standard_audio_file(chapter_files):
    """
    Uses ffmpeg to merge all chapter files into a standard M4A audio file).

    This function takes a list of chapter files and an output format as input, and generates a standard M4A audio file.

    Args:
        chapter_files (list): A list of the paths to the individual chapter audio files.
    """
    file_list_path = "chapter_list.txt"
    
    # Write the list of chapter files to a text file (ffmpeg input)
    with open(file_list_path, "w", encoding='utf-8') as f:
        for chapter in chapter_files:
            chapter_path = os.path.join('temp_audio', chapter)
            # Validate each chapter file
            if not validate_file_path(chapter_path):
                raise ValueError(f"Invalid chapter file: {chapter}")
            f.write(f"file '{chapter_path}'\n")

    # Construct the output file path
    output_file = "generated_audiobooks/audiobook.m4a"

    # Validate file list exists
    if not validate_file_path(file_list_path):
        raise ValueError("Chapter list file is invalid")

    # Construct secure ffmpeg command
    ffmpeg_cmd = [
        "ffmpeg", "-y", 
        "-f", "concat", "-safe", "0", 
        "-i", file_list_path, 
        "-c", "copy", 
        output_file
    ]

    # Use centralized secure command execution
    allowed_ffmpeg_commands = ['ffmpeg']
    result = run_shell_command_secure(ffmpeg_cmd, allowed_ffmpeg_commands)
    
    if not result or result.returncode != 0:
        error_msg = result.stderr if result else "Unknown error"
        raise RuntimeError(f"FFmpeg failed: {error_msg}")
        
    print(f"Audiobook created: {output_file}")

def assemble_chapter_with_ffmpeg(chapter_file, line_indices, temp_line_audio_dir, temp_audio_dir):
    """
    Memory-efficient chapter assembly using FFmpeg instead of PyDub.
    
    This function concatenates line audio files into a chapter using FFmpeg's concat filter,
    which is much more memory-efficient than loading all audio into PyDub AudioSegments.
    
    Args:
        chapter_file (str): The name of the chapter file to create
        line_indices (list): List of line indices that belong to this chapter
        temp_line_audio_dir (str): Directory containing the line audio files
        temp_audio_dir (str): Directory where the chapter file will be created
    """
    if not line_indices:
        return
    
    # Create a temporary file list for FFmpeg concat
    file_list_path = os.path.join(temp_audio_dir, f"chapter_list_{chapter_file}.txt")
    
    try:
        with open(file_list_path, "w", encoding='utf-8') as f:
            for line_index in sorted(line_indices):
                line_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.wav")
                # Use absolute path to avoid issues with FFmpeg concat
                abs_line_path = os.path.abspath(line_path)
                f.write(f"file '{abs_line_path}'\n")
        
        # Create the full chapter file path
        chapter_path = os.path.join(temp_audio_dir, chapter_file)
        
        # Use FFmpeg to concatenate the audio files
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", file_list_path, "-c", "copy", chapter_path
        ]
        
        # Use centralized secure command execution
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(ffmpeg_cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            raise RuntimeError(f"FFmpeg chapter assembly failed: {error_msg}")
        
    finally:
        # Clean up the temporary file list
        if os.path.exists(file_list_path):
            os.unlink(file_list_path)

def get_audio_properties(file_path):
    """
    Extract audio properties from an audio file using ffprobe.
    
    This function analyzes an audio file and returns its sample rate, channel count,
    and channel layout, which is useful for ensuring compatibility when processing audio.
    
    Args:
        file_path (str): Path to the audio file to analyze
        
    Returns:
        dict: Dictionary containing:
            - sample_rate (int): Sample rate in Hz
            - channels (int): Number of audio channels  
            - channel_layout (str): Channel layout ("mono" or "stereo")
            - duration (float): Duration in seconds (bonus info)
    """
    # Validate file path
    if not validate_file_path(file_path):
        # Return fallback values for invalid paths instead of raising exception
        # since this function is often used as a helper
        return {
            "sample_rate": 44100,
            "channels": 1,
            "channel_layout": "mono",
            "duration": 0.0
        }
    
    # Get audio properties of the file
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", 
        "-show_streams", "-show_format", file_path
    ]
    
    try:
        # Use centralized secure command execution
        allowed_commands = ['ffprobe']
        probe_result = run_shell_command_secure(probe_cmd, allowed_commands)
        
        if not probe_result or probe_result.returncode != 0:
            raise ValueError("ffprobe failed")
            
        probe_data = json.loads(probe_result.stdout)
        
        # Extract audio stream properties
        audio_stream = next((s for s in probe_data["streams"] if s["codec_type"] == "audio"), None)
        if audio_stream:
            sample_rate = int(audio_stream["sample_rate"])
            channels = int(audio_stream["channels"])
            channel_layout = "mono" if channels == 1 else "stereo"
            
            # Get duration from format info
            duration = float(probe_data["format"]["duration"]) if "format" in probe_data and "duration" in probe_data["format"] else 0.0
            
            return {
                "sample_rate": sample_rate,
                "channels": channels,
                "channel_layout": channel_layout,
                "duration": duration
            }
        else:
            # No audio stream found, return fallback values
            return {
                "sample_rate": 44100,
                "channels": 1,
                "channel_layout": "mono",
                "duration": 0.0
            }
            
    except (ValueError, json.JSONDecodeError, KeyError):
        # Fallback to common values if probe fails
        return {
            "sample_rate": 44100,
            "channels": 1,
            "channel_layout": "mono",
            "duration": 0.0
        }

def add_silence_to_chapter_with_ffmpeg(chapter_path, silence_duration_ms=1000):
    """
    Memory-efficient silence addition using FFmpeg instead of PyDub.
    
    This function adds silence to the end of a chapter file using FFmpeg,
    which is much more memory-efficient than loading the entire chapter into PyDub.
    
    Args:
        chapter_path (str): Path to the chapter audio file
        silence_duration_ms (int): Duration of silence to add in milliseconds (default: 1000ms = 1 second)
    """
    # Validate chapter path
    if not validate_file_path(chapter_path):
        raise ValueError(f"Invalid chapter path: {chapter_path}")
        
    # Get audio properties of the chapter file
    audio_props = get_audio_properties(chapter_path)
    sample_rate = audio_props["sample_rate"]
    channel_layout = audio_props["channel_layout"]
    
    # Use sample-rate specific silence file
    silence_file_path = f"static_files/silence_{sample_rate}hz.wav"
    
    # Check if silence file exists, if not create it
    if not os.path.exists(silence_file_path):
        generate_silence_file(silence_file_path, silence_duration_ms, sample_rate, channel_layout)
    
    # Create temporary output path with proper extension
    temp_output_path = f"{chapter_path}.temp.wav"
    
    # Create temporary concat list file
    concat_list_path = f"{chapter_path}.concat_list.txt"
    
    try:
        # Create a concat list file for FFmpeg
        with open(concat_list_path, "w", encoding='utf-8') as f:
            f.write(f"file '{os.path.abspath(chapter_path)}'\n")
            f.write(f"file '{os.path.abspath(silence_file_path)}'\n")
        
        # Use FFmpeg concat demuxer to join the files with lossless copy
        ffmpeg_cmd = [
            "ffmpeg", "-y", 
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",  # LOSSLESS: Copy without re-encoding
            temp_output_path
        ]
        
        # Use centralized secure command execution
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(ffmpeg_cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            raise RuntimeError(f"FFmpeg silence addition failed: {error_msg}")
        
        # Replace the original file with the new one
        os.replace(temp_output_path, chapter_path)
        
    except Exception as e:
        # Clean up temp files if FFmpeg failed
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)
        print(f"FFmpeg error: {e}")
        raise e
    finally:
        # Clean up concat list file
        if os.path.exists(concat_list_path):
            os.unlink(concat_list_path)

def generate_silence_file(output_path, duration_ms=1000, sample_rate=44100, channel_layout="mono"):
    """
    Generate a silence audio file using FFmpeg with specific audio properties.
    
    This function creates a WAV file containing silence that matches the sample rate
    and channel layout of the TTS-generated audio for perfect compatibility.
    
    Args:
        output_path (str): Path where the silence file will be saved
        duration_ms (int): Duration of silence in milliseconds (default: 1000ms = 1 second)
        sample_rate (int): Sample rate in Hz (default: 44100)
        channel_layout (str): Channel layout - "mono" or "stereo" (default: "mono")
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert milliseconds to seconds for FFmpeg
    duration_seconds = duration_ms / 1000.0
    
    # Generate silence using FFmpeg with matching properties
    ffmpeg_cmd = [
        "ffmpeg", "-y", 
        "-f", "lavfi", 
        "-i", f"anullsrc=r={sample_rate}:cl={channel_layout}",
        "-t", str(duration_seconds),
        "-c:a", "pcm_s16le",  # WAV format
        "-ar", str(sample_rate),  # Match sample rate
        output_path
    ]
    
    try:
        # Use centralized secure command execution
        allowed_commands = ['ffmpeg']
        result = run_shell_command_secure(ffmpeg_cmd, allowed_commands)
        
        if not result or result.returncode != 0:
            error_msg = result.stderr if result else "Unknown error"
            raise RuntimeError(f"Failed to generate silence file: {error_msg}")
            
        print(f"Generated silence file: {output_path} ({duration_ms}ms, {sample_rate}Hz, {channel_layout})")
    except Exception as e:
        print(f"Error generating silence file: {e}")
        raise e
