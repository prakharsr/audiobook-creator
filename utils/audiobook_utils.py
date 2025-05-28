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
from config.constants import CHAPTER_LIST_FILE, FFMPEG_METADATA_FILE, TEMP_DIR
from utils.run_shell_commands import run_shell_command


# Escape double quotes by replacing them with \"
def escape_metadata(value):
    if value:
        return value.replace('"', '\\"')  # Escape double quotes
    return ""


def get_ebook_metadata_with_cover(book_path, book_title="audiobook"):
    """
    Extracts metadata from an ebook and saves its cover image.

    Args:
        book_path (str): The path to the ebook file.

    Returns:
        dict: A dictionary containing the ebook's metadata.
    """
    cover_path = f"{TEMP_DIR}/{book_title}/cover.jpg"
    try:
        # First try to find ebook-meta in PATH
        ebook_meta_bin_result = run_shell_command("which ebook-meta")

        ebook_meta_bin_path = None

        # Check if ebook-meta was found in PATH
        if ebook_meta_bin_result and ebook_meta_bin_result.stdout.strip():
            ebook_meta_bin_path = ebook_meta_bin_result.stdout.strip()
        else:
            # Try common installation paths for Calibre
            common_paths = [
                "/Applications/calibre.app/Contents/MacOS/ebook-meta",  # macOS App Store/DMG install
                "/usr/local/bin/ebook-meta",  # Homebrew install
                "/opt/homebrew/bin/ebook-meta",  # Homebrew Apple Silicon
                "/usr/bin/ebook-meta",  # System install
            ]

            for path in common_paths:
                if os.path.exists(path):
                    ebook_meta_bin_path = path
                    break

        if not ebook_meta_bin_path:
            print("Warning: ebook-meta not found. Using default metadata for M4B file.")

            # Return default metadata when ebook-meta is not available
            return {
                "Title": book_title,
                "Author(s)": "Unknown",
                "Publisher": "Unknown",
                "Languages": "en",
                "Published": "Unknown",
                "Comments": "Generated audiobook",
            }

        # Command to extract metadata and cover image using ebook-meta
        command = f"'{ebook_meta_bin_path}' '{book_path}' --get-cover '{cover_path}'"

        # Run the command and capture the result
        result = run_shell_command(command)

        if result is None:
            print(
                "Warning: Failed to extract metadata. Using default metadata for M4B file."
            )
            return {
                "Title": book_title,
                "Author(s)": "Unknown",
                "Publisher": "Unknown",
                "Languages": "en",
                "Published": "Unknown",
                "Comments": "Generated audiobook",
            }

        metadata = {}
        # Parse the command output to extract metadata
        for line in result.stdout.split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                metadata[key.strip()] = value.strip()

        # Check if cover was actually extracted
        cover_path = f"{TEMP_DIR}/{book_title}/cover.jpg"
        if os.path.exists(cover_path):
            print(f"✅ Cover image extracted from epub: {cover_path}")
        else:
            print(f"Warning: Cover image was not extracted.")

        return metadata

    except Exception as e:
        print(
            f"Warning: Error extracting metadata: {e}. Using default metadata for M4B file."
        )

        # Return default metadata when there's an error
        return {
            "Title": book_title,
            "Author(s)": "Unknown",
            "Publisher": "Unknown",
            "Languages": "en",
            "Published": "Unknown",
            "Comments": "Generated audiobook",
        }


def get_audio_duration_using_ffprobe(file_path):
    """
    Returns the duration of an audio file in milliseconds using ffprobe.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        int: The duration of the audio file in milliseconds.
    """
    # Construct the command to execute
    cmd = [
        "ffprobe",  # Use ffprobe to get the duration
        "-v",
        "error",  # Set the verbosity to error
        "-show_entries",  # Show the specified entries
        "format=duration",  # Show the duration
        "-of",  # Specify the output format
        "default=noprint_wrappers=1:nokey=1",  # Print the duration without any additional information
        file_path,  # Specify the file to analyze
    ]
    # Run the command and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Convert the output to an integer (in milliseconds) and return it
    return int(float(result.stdout.strip()) * 1000)


def get_audio_duration_using_raw_ffmpeg(file_path):
    """Returns the duration of an audio file using FFmpeg."""
    cmd = ["ffmpeg", "-y", "-i", file_path, "-f", "null", "-"]

    try:
        result = subprocess.run(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
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


def generate_chapters_file(
    chapter_files, book_title="audiobook", output_file=FFMPEG_METADATA_FILE
):
    """
    Generates a chapter metadata file for FFmpeg.

    The chapter metadata file is a text file that contains information about each chapter in the audiobook, such as the chapter title and the start and end times of the chapter.

    Args:
        chapter_files (list): A list of the paths to the individual chapter audio files.
        book_title (str): The title/name of the book for directory structure.
        output_file (str): The path to the output chapter metadata file. Defaults to "chapters.txt".
    """
    start_time = 0
    output_metadata_path = os.path.join(TEMP_DIR, book_title, output_file)

    with open(output_metadata_path, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for chapter in chapter_files:
            chapter_path = os.path.join(TEMP_DIR, book_title, chapter)
            if not os.path.exists(chapter_path):
                raise FileNotFoundError(f"Chapter file not found: {chapter_path}")

            try:
                duration = get_audio_duration_using_ffprobe(chapter_path)
            except Exception as e:
                raise RuntimeError(f"Failed to get duration for {chapter}: {e}")

            end_time = start_time + duration
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_time}\n")
            f.write(f"END={end_time}\n")
            f.write(f"title={os.path.splitext(chapter)[0]}\n\n")
            start_time = end_time


def create_m4a_file_from_raw_aac_file(input_file_path, output_file_path):
    cmd = ["ffmpeg", "-y", "-i", input_file_path, "-c", "copy", output_file_path]

    try:
        result = subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def create_m4a_file_from_wav_file(input_file_path, output_file_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        "-c:a",
        "aac",
        "-b:a",
        "240k",
        output_file_path,
    ]

    try:
        result = subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def create_aac_file_from_m4a_file(input_file_path, output_file_path):
    cmd = ["ffmpeg", "-y", "-i", input_file_path, "-c", "copy", output_file_path]

    try:
        result = subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def create_mp3_file_from_m4a_file(input_file_path, output_file_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        output_file_path,
    ]

    try:
        result = subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def create_wav_file_from_m4a_file(input_file_path, output_file_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        "-c:a",
        "pcm_s16le",
        "-ar",
        "44100",
        output_file_path,
    ]

    try:
        result = subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def create_opus_file_from_m4a_file(input_file_path, output_file_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        "-c:a",
        "libopus",
        "-b:a",
        "128k",
        output_file_path,
    ]

    try:
        result = subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def create_flac_file_from_m4a_file(input_file_path, output_file_path):
    cmd = ["ffmpeg", "-y", "-i", input_file_path, "-c:a", "flac", output_file_path]

    try:
        result = subprocess.run(cmd)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def create_pcm_file_from_m4a_file(input_file_path, output_file_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        output_file_path,
    ]

    try:
        result = subprocess.run(cmd)
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
            pass  # Already generated
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


def merge_chapters_to_m4b(book_path, chapter_files, book_title="audiobook"):
    """
    Uses ffmpeg to merge all chapter files into an M4B audiobook.

    This function takes the path to the book file and a list of chapter files as input, and generates an M4B audiobook with chapter metadata and a cover image.

    Args:
        book_path (str): The path to the book file.
        chapter_files (list): A list of the paths to the individual chapter audio files.
        book_title (str): The title/name of the book for directory structure.
    """
    file_list_path = f"{TEMP_DIR}/{book_title}/{CHAPTER_LIST_FILE}"
    print(file_list_path)

    with open(file_list_path, "w", encoding="utf-8") as f:
        for chapter in chapter_files:
            print(chapter)
            f.write(f"file '{chapter}'\n")

    metadata = get_ebook_metadata_with_cover(book_path, book_title)
    title = escape_metadata(metadata.get("Title", ""))
    authors = escape_metadata(metadata.get("Author(s)", ""))
    publisher = escape_metadata(metadata.get("Publisher", ""))
    languages = escape_metadata(metadata.get("Languages", ""))
    published_date = escape_metadata(metadata.get("Published", ""))
    comments = escape_metadata(metadata.get("Comments", ""))

    # Generate chapter metadata
    generate_chapters_file(chapter_files, book_title, FFMPEG_METADATA_FILE)

    output_m4b = f"generated_audiobooks/{book_title}.m4b"
    cover_image = f"{TEMP_DIR}/{book_title}/cover.jpg"
    chapters_file = f"{TEMP_DIR}/{book_title}/{FFMPEG_METADATA_FILE}"

    # Construct metadata arguments safely
    metadata = (
        f'-metadata title="{title}" '
        f'-metadata artist="{authors}" '
        f'-metadata album="{title}" '
        f'-metadata genre="Audiobook" '
        f'-metadata publisher="{publisher}" '
        f'-metadata language="{languages}" '
        f'-metadata date="{published_date}" '
        f'-metadata description="{comments}"'
    )

    # Optimized FFmpeg command with threading and better performance
    ffmpeg_cmd = (
        f'ffmpeg -y -f concat -safe 0 -i "{file_list_path}" -i "{cover_image}" -i "{chapters_file}" '
        f"-c:a copy -c:v copy -map 0:a -map 1:v -disposition:v:0 attached_pic -map_metadata 2 "
        f'-avoid_negative_ts make_zero -fflags +genpts -threads 0 {metadata} "{output_m4b}"'
    )

    try:
        # Try with copy codec first (fastest)
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Fallback to re-encoding if copy fails
        print("Copy codec failed for M4B, falling back to re-encoding...")
        ffmpeg_cmd = (
            f'ffmpeg -y -f concat -safe 0 -i {file_list_path} -i "{cover_image}" -i "{chapters_file}" '
            f"-c:a aac -b:a 256k -c:v copy -map 0:a -map 1:v -disposition:v:0 attached_pic -map_metadata 2 "
            f'-avoid_negative_ts make_zero -fflags +genpts -threads 0 {metadata} "{output_m4b}"'
        )
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
    print(f"Audiobook created: {output_m4b}")


def add_silence_to_audio_file_by_appending_silence_file(input_file_path):
    silence_path = "static_files/silence.aac"  # Pre generated 1 seconds of silence using command `ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -c:a aac silence.aac`

    with open(silence_path, "rb") as silence_file, open(
        input_file_path, "ab"
    ) as audio_file:
        audio_file.write(
            silence_file.read()
        )  # Append silence to the end of the audio file


def add_silence_to_audio_file_by_reencoding_using_ffmpeg(
    temp_dir, input_file_name, pause_duration
):
    """
    Adds a silence of specified duration at the end of an audio file.

    Args:
        temp_dir (str): The temporary directory to store the silence file.
        input_file_name (str): The name of the file to add silence to.
        pause_duration (str): The duration of the silence (e.g. 00:00:05).
    """
    # Generate a silence file with the specified duration
    generate_silence_command = f'ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=mono -t {pause_duration} -c:a aac "{temp_dir}/silence.aac"'
    subprocess.run(generate_silence_command, shell=True, check=True)

    # Add the silence to the end of the audio file
    add_silence_command = f'ffmpeg -y -i "{temp_dir}/{input_file_name}" -i "{temp_dir}/silence.aac" -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" "{temp_dir}/temp_audio_file.aac"'
    subprocess.run(add_silence_command, shell=True, check=True)

    # Rename the temporary file back to the original file name
    rename_file_command = (
        f'mv "{temp_dir}/temp_audio_file.aac" "{temp_dir}/{input_file_name}"'
    )
    subprocess.run(rename_file_command, shell=True, check=True)


def merge_chapters_to_standard_audio_file(chapter_files, book_title="audiobook"):
    """
    Uses ffmpeg to merge all chapter files into a standard M4A audio file).

    This function takes a list of chapter files and an output format as input, and generates a standard M4A audio file.

    Args:
        chapter_files (list): A list of the paths to the individual chapter audio files.
        book_title (str): The title/name of the book for directory structure.
    """
    file_list_path = f"{TEMP_DIR}/{book_title}/{CHAPTER_LIST_FILE}"

    # Write the list of chapter files to a text file (ffmpeg input)
    with open(file_list_path, "w", encoding="utf-8") as f:
        for chapter in chapter_files:
            # Use absolute path for each chapter file
            abs_chapter_path = os.path.abspath(
                os.path.join(TEMP_DIR, book_title, chapter)
            )
            f.write(f"file '{abs_chapter_path}'\n")

    # Use sanitized book title for filename to avoid issues with spaces
    safe_book_title = "".join(
        c for c in book_title if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    safe_book_title = (
        safe_book_title.replace(" ", "_") or "audiobook"
    )  # Replace spaces with underscores

    # Construct the output file path
    output_file = f"generated_audiobooks/{safe_book_title}.m4a"

    # Use optimized FFmpeg parameters for faster processing
    # Use copy codec when possible to avoid re-encoding, add threading
    ffmpeg_cmd = f'ffmpeg -y -f concat -safe 0 -i "{file_list_path}" -c:a copy -avoid_negative_ts make_zero -fflags +genpts -threads 0 "{output_file}"'

    try:
        # Run the ffmpeg command with copy codec first (fastest)
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Fallback to re-encoding if copy fails
        print("Copy codec failed, falling back to re-encoding...")
        ffmpeg_cmd = f'ffmpeg -y -f concat -safe 0 -i "{file_list_path}" -c:a aac -b:a 256k -avoid_negative_ts make_zero -fflags +genpts -threads 0 "{output_file}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True)

    # Print a message when the generation is complete
    print(f"Audiobook created: {output_file}")
