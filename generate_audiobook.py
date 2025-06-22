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

import shutil
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import json
import os
import asyncio
import re
from word2number import w2n
import time
import sys
from utils.run_shell_commands import check_if_ffmpeg_is_installed, check_if_calibre_is_installed
from utils.file_utils import read_json, empty_directory
from utils.audiobook_utils import merge_chapters_to_m4b, convert_audio_file_formats, add_silence_to_audio_file_by_reencoding_using_ffmpeg, merge_chapters_to_standard_audio_file, add_silence_to_audio_file_by_appending_pre_generated_silence
from utils.check_if_kokoro_api_is_up import check_if_kokoro_api_is_up
from dotenv import load_dotenv

load_dotenv()

KOKORO_BASE_URL = os.environ.get("KOKORO_BASE_URL", "http://localhost:8880/v1")
KOKORO_API_KEY = os.environ.get("KOKORO_API_KEY", "not-needed")
KOKORO_MAX_PARALLEL_REQUESTS_BATCH_SIZE = int(os.environ.get("KOKORO_MAX_PARALLEL_REQUESTS_BATCH_SIZE", 2))

os.makedirs("audio_samples", exist_ok=True)
os.makedirs("generated_audiobooks", exist_ok=True)

async_openai_client = AsyncOpenAI(
    base_url=KOKORO_BASE_URL, api_key=KOKORO_API_KEY
)

def sanitize_filename(text):
    # Remove or replace problematic characters
    text = text.replace("'", '').replace('"', '').replace('/', ' ').replace('.', ' ')
    text = text.replace(':', '').replace('?', '').replace('\\', '').replace('|', '')
    text = text.replace('*', '').replace('<', '').replace('>', '').replace('&', 'and')
    
    # Normalize whitespace and trim
    text = ' '.join(text.split())
    
    return text

def split_and_annotate_text(text):
    """Splits text into dialogue and narration while annotating each segment."""
    parts = re.split(r'("[^"]+")', text)  # Keep dialogues in the split result
    annotated_parts = []

    for part in parts:
        if part:  # Ignore empty strings
            annotated_parts.append({
                "text": part,
                "type": "dialogue" if part.startswith('"') and part.endswith('"') else "narration"
            })

    return annotated_parts

def check_if_chapter_heading(text):
    """
    Checks if a given text line represents a chapter heading.

    A chapter heading is considered a string that starts with either "Chapter",
    "Part", or "PART" (case-insensitive) followed by a number (either a digit
    or a word that can be converted to an integer).

    :param text: The text to check
    :return: True if the text is a chapter heading, False otherwise
    """
    pattern = r'^(Chapter|Part|PART)\s+([\w-]+|\d+)'
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.match(text)

    if match:
        label, number = match.groups()
        try:
            # Try converting the number (either digit or word) to an integer
            w2n.word_to_num(number) if not number.isdigit() else int(number)
            return True
        except ValueError:
            return False  # Invalid number format
    return False  # No match
    
def find_voice_for_gender_score(character: str, character_gender_map, kokoro_voice_map):
    """
    Finds the appropriate voice for a character based on their gender score.

    This function takes in the name of a character, a dictionary mapping character names to their gender scores,
    and a dictionary mapping voice identifiers to gender scores. It returns the voice identifier that matches the
    character's gender score.

    Args:
        character (str): The name of the character for whom the voice is being determined.
        character_gender_map (dict): A dictionary mapping character names to their gender scores.
        kokoro_voice_map (dict): A dictionary mapping voice identifiers to gender scores.

    Returns:
        str: The voice identifier that matches the character's gender score.
    """

    # Get the character's gender score
    character_gender_score_doc = character_gender_map["scores"][character.lower()]
    character_gender_score = character_gender_score_doc["gender_score"]

    # Iterate over the voice identifiers and their scores
    for voice, score in kokoro_voice_map.items():
        # Find the voice identifier that matches the character's gender score
        if score == character_gender_score:
            return voice

async def generate_audio_with_single_voice(output_format, narrator_gender, generate_m4b_audiobook_file=False, book_path=""):
    # Read the text from the file
    """
    Generate an audiobook using a single voice for narration and dialogues.

    This asynchronous function reads text from a file, processes each line to determine
    if it is narration or dialogue, and generates corresponding audio using specified
    voices. The generated audio is organized by chapters, with options to create
    an M4B audiobook file or a standard audio file in the specified output format.

    Args:
        output_format (str): The desired output format for the final audiobook (e.g., "mp3", "wav").
        narrator_gender (str): The gender of the narrator ("male" or "female") to select appropriate voices.
        generate_m4b_audiobook_file (bool, optional): Flag to determine whether to generate an M4B file. Defaults to False.
        book_path (str, optional): The file path for the book to be used in M4B creation. Defaults to an empty string.

    Yields:
        str: Progress updates as the audiobook generation progresses through loading text, generating audio,
             organizing by chapters, assembling chapters, and post-processing steps.
    """

    with open("converted_book.txt", "r", encoding='utf-8') as f:
        text = f.read()
    lines = text.split("\n")
    
    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Set the voices to be used
    narrator_voice = "" # voice to be used for narration
    dialogue_voice = "" # voice to be used for dialogue

    if narrator_gender == "male":
        narrator_voice = "am_puck"
        dialogue_voice = "af_alloy+am_puck"
    else:
        narrator_voice = "af_heart"
        dialogue_voice = "af_sky"

    # Setup directories
    temp_audio_dir = "temp_audio"
    temp_line_audio_dir = os.path.join(temp_audio_dir, "line_segments")

    empty_directory(temp_audio_dir)

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)
    
    # Batch processing parameters
    semaphore = asyncio.Semaphore(KOKORO_MAX_PARALLEL_REQUESTS_BATCH_SIZE)
    
    # Initial setup for chapters
    chapter_index = 1
    current_chapter_audio = f"Introduction.aac"
    chapter_files = []
    
    # First pass: Generate audio for each line independently
    total_size = len(lines)

    progress_counter = 0
    
    # For tracking progress with tqdm in an async context
    progress_bar = tqdm(total=total_size, unit="line", desc="Audio Generation Progress")
    
    # Maps chapters to their line indices
    chapter_line_map = {}
    
    async def process_single_line(line_index, line):
        async with semaphore:
            nonlocal progress_counter

            if not line:
                return None
                
            # Split the line into annotated parts
            annotated_parts = split_and_annotate_text(line)
            
            # Create an in-memory buffer for the audio data
            audio_buffer = bytearray()
            
            for part in annotated_parts:
                text_to_speak = part["text"]
                voice_to_speak_in = narrator_voice if part["type"] == "narration" else dialogue_voice
                
                # Generate audio for the part
                async with async_openai_client.audio.speech.with_streaming_response.create(
                    model="kokoro",
                    voice=voice_to_speak_in,
                    response_format="aac",
                    speed=0.85,
                    input=text_to_speak
                ) as response:
                    async for chunk in response.iter_bytes():
                        audio_buffer.extend(chunk)
            
            # Write this line's audio to a temporary file
            line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.aac")
            with open(line_audio_path, "wb") as line_file:
                line_file.write(audio_buffer)
            
            # Update progress bar
            progress_bar.update(1)
            progress_counter += 1
            
            return {
                "index": line_index,
                "is_chapter_heading": check_if_chapter_heading(line),
                "line": line,
            }

    # Create tasks and store them with their index for result collection
    tasks = []
    task_to_index = {}
    for i, line in enumerate(lines):
        task = asyncio.create_task(process_single_line(i, line))
        tasks.append(task)
        task_to_index[task] = i
    
    # Initialize results_all list
    results_all = [None] * len(lines)
    
    # Process tasks with progress updates
    last_reported = -1
    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Store results as tasks complete
        for completed_task in done:
            idx = task_to_index[completed_task]
            results_all[idx] = completed_task.result()
        
        tasks = list(pending)
        
        # Only yield if the counter has changed
        if progress_counter > last_reported:
            last_reported = progress_counter
            percent = (progress_counter / total_size) * 100
            yield f"Generating audiobook. Progress: {percent:.1f}%"
    
    # All tasks have completed at this point and results_all is populated
    results = [r for r in results_all if r is not None]  # Filter out empty lines
    
    progress_bar.close()
    
    # Filter out empty lines (same as in your original code)
    results = [r for r in results_all if r is not None]
    
    yield "Completed generating audio for all lines"

    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(total=len(results), unit="result", desc="Organizing Chapters")
    
    for result in sorted(results, key=lambda x: x["index"]):
        # Check if this is a chapter heading
        if result["is_chapter_heading"]:
            chapter_index += 1
            current_chapter_audio = f"{sanitize_filename(result['line'])}.aac"
            
        if current_chapter_audio not in chapter_files:
            chapter_files.append(current_chapter_audio)
            chapter_line_map[current_chapter_audio] = []
            
        # Add this line index to the chapter
        chapter_line_map[current_chapter_audio].append(result["index"])
        chapter_organization_bar.update(1)
    
    chapter_organization_bar.close()
    yield "Organizing audio by chapters complete"
    
    # Third pass: Concatenate audio files for each chapter in order
    chapter_assembly_bar = tqdm(total=len(chapter_files), unit="chapter", desc="Assembling Chapters")
    
    for chapter_file in chapter_files:
        chapter_path = os.path.join(temp_audio_dir, chapter_file)
        
        with open(chapter_path, "wb") as chapter_output:
            # Create progress bar for lines in this chapter
            line_assembly_bar = tqdm(
                total=len(chapter_line_map[chapter_file]), 
                unit="line", 
                desc=f"Assembling {chapter_file[:20]}..."
            )
            
            # Process each line in the correct order
            for line_index in sorted(chapter_line_map[chapter_file]):
                line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.aac")
                
                # Read the line audio and append it to the chapter file
                with open(line_audio_path, "rb") as line_input:
                    chapter_output.write(line_input.read())
                
                line_assembly_bar.update(1)
            
            line_assembly_bar.close()
        
        chapter_assembly_bar.update(1)
        yield f"Assembled chapter: {chapter_file}"
    
    chapter_assembly_bar.close()
    
    # Post-processing steps
    post_processing_bar = tqdm(total=len(chapter_files)*2, unit="task", desc="Post Processing")
    
    # Add silence to each chapter file
    for chapter in chapter_files:
        add_silence_to_audio_file_by_appending_pre_generated_silence(temp_audio_dir, chapter)
        post_processing_bar.update(1)
        yield f"Added silence to chapter: {chapter}"

    m4a_chapter_files = []

    # Convert all chapter files to M4A format
    for chapter in chapter_files:
        chapter_name = chapter.split('.')[0]
        m4a_chapter_files.append(f"{chapter_name}.m4a")
        # Convert to M4A as raw AAC have problems with timestamps and metadata
        convert_audio_file_formats("aac", "m4a", temp_audio_dir, chapter_name)
        post_processing_bar.update(1)
        yield f"Converted chapter to M4A: {chapter_name}"
    
    post_processing_bar.close()
    
    # Clean up temp line audio files
    shutil.rmtree(temp_line_audio_dir)
    yield "Cleaned up temporary files"

    if generate_m4b_audiobook_file:
        # Merge all chapter files into a final m4b audiobook
        yield "Creating M4B audiobook file..."
        merge_chapters_to_m4b(book_path, m4a_chapter_files)
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files)
        convert_audio_file_formats("m4a", output_format, "generated_audiobooks", "audiobook")
        yield f"Audiobook in {output_format} format created successfully"

async def generate_audio_with_multiple_voices(output_format, narrator_gender, generate_m4b_audiobook_file=False, book_path=""):
    # Path to the JSONL file containing speaker-attributed lines
    """
    Generate an audiobook in the specified format using multiple voices for each line

    Uses the provided JSONL file to map speaker names to voices. The JSONL file should contain
    entries with the following format:
    {
        "line": <string>,
        "speaker": <string>
    }

    The function will generate audio for each line independently and then concatenate the audio
    files for each chapter in order. The final audiobook will be saved in the "generated_audiobooks"
    directory with the name "audiobook.<format>".

    :param output_format: The desired format of the final audiobook (e.g. "m4a", "mp3")
    :param narrator_gender: The gender of the narrator voice (e.g. "male", "female")
    :param generate_m4b_audiobook_file: Whether to generate an M4B audiobook file instead of a standard
    M4A file
    :param book_path: The path to the book file (required for generating an M4B audiobook file)
    """
    file_path = 'speaker_attributed_book.jsonl'
    json_data_array = []

    # Open the JSONL file and read it line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            json_object = json.loads(line.strip())
            # Append the parsed JSON object to the array
            json_data_array.append(json_object)

    yield "Loaded speaker-attributed lines from JSONL file"

    # Load mappings for character gender and voice selection
    character_gender_map = read_json("character_gender_map.json")
    kokoro_voice_map = None

    if narrator_gender == "male":
        kokoro_voice_map = read_json("static_files/kokoro_voice_map_male_narrator.json")
    else:
        kokoro_voice_map = read_json("static_files/kokoro_voice_map_female_narrator.json")

    narrator_voice = find_voice_for_gender_score("narrator", character_gender_map, kokoro_voice_map)
    yield "Loaded voice mappings and selected narrator voice"
    
    # Setup directories
    temp_audio_dir = "temp_audio"
    temp_line_audio_dir = os.path.join(temp_audio_dir, "line_segments")

    empty_directory(temp_audio_dir)

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)
    yield "Set up temporary directories for audio processing"
    
    # Batch processing parameters
    semaphore = asyncio.Semaphore(KOKORO_MAX_PARALLEL_REQUESTS_BATCH_SIZE)
    
    # Initial setup for chapters
    chapter_index = 1
    current_chapter_audio = f"Introduction.aac"
    chapter_files = []
    
    # First pass: Generate audio for each line independently
    # and track chapter organization
    chapter_line_map = {}  # Maps chapters to their line indices

    progress_counter = 0
    
    # For tracking progress with tqdm in an async context
    total_lines = len(json_data_array)
    progress_bar = tqdm(total=total_lines, unit="line", desc="Audio Generation Progress")

    yield "Generating audio..."

    async def process_single_line(line_index, doc):
        async with semaphore:
            nonlocal progress_counter

            line = doc["line"].strip()
            if not line:
                progress_bar.update(1)  # Update the progress bar even for empty lines
                return None
                
            speaker = doc["speaker"]
            speaker_voice = find_voice_for_gender_score(speaker, character_gender_map, kokoro_voice_map)
            
            # Split the line into annotated parts
            annotated_parts = split_and_annotate_text(line)
            
            # Create an in-memory buffer for the audio data
            audio_buffer = bytearray()
            
            for part in annotated_parts:
                text_to_speak = part["text"]
                voice_to_speak_in = narrator_voice if part["type"] == "narration" else speaker_voice
                
                # Generate audio for the part
                async with async_openai_client.audio.speech.with_streaming_response.create(
                    model="kokoro",
                    voice=voice_to_speak_in,
                    response_format="aac",
                    speed=0.85,
                    input=text_to_speak
                ) as response:
                    async for chunk in response.iter_bytes():
                        audio_buffer.extend(chunk)
            
            # Write this line's audio to a temporary file
            line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.aac")
            with open(line_audio_path, "wb") as line_file:
                line_file.write(audio_buffer)
            
            # Update progress bar
            progress_bar.update(1)
            progress_counter += 1
            
            return {
                "index": line_index,
                "is_chapter_heading": check_if_chapter_heading(line),
                "line": line
            }
    
    # Create tasks and store them with their index for result collection
    tasks = []
    task_to_index = {}
    for i, doc in enumerate(json_data_array):
        task = asyncio.create_task(process_single_line(i, doc))
        tasks.append(task)
        task_to_index[task] = i
    
    # Initialize results_all list
    results_all = [None] * len(json_data_array)
    
    # Process tasks with progress updates
    last_reported = -1
    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Store results as tasks complete
        for completed_task in done:
            idx = task_to_index[completed_task]
            results_all[idx] = completed_task.result()
        
        tasks = list(pending)
        
        # Only yield if the counter has changed
        if progress_counter > last_reported:
            last_reported = progress_counter
            percent = (progress_counter / total_lines) * 100
            yield f"Generating audiobook. Progress: {percent:.1f}%"
    
    # All tasks have completed at this point and results_all is populated
    results = [r for r in results_all if r is not None]  # Filter out empty lines
    
    progress_bar.close()
    
    # Filter out empty lines (same as in your original code)
    results = [r for r in results_all if r is not None]
    
    yield "Completed generating audio for all lines"
    
    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(total=len(results), unit="result", desc="Organizing Chapters")
    yield "Organizing lines into chapters"
    
    for result in sorted(results, key=lambda x: x["index"]):
        # Check if this is a chapter heading
        if result["is_chapter_heading"]:
            chapter_index += 1
            current_chapter_audio = f"{sanitize_filename(result['line'])}.aac"
            
        if current_chapter_audio not in chapter_files:
            chapter_files.append(current_chapter_audio)
            chapter_line_map[current_chapter_audio] = []
            
        # Add this line index to the chapter
        chapter_line_map[current_chapter_audio].append(result["index"])
        chapter_organization_bar.update(1)
    
    chapter_organization_bar.close()
    yield f"Organized {len(results)} lines into {len(chapter_files)} chapters"
    
    # Third pass: Concatenate audio files for each chapter in order
    chapter_assembly_bar = tqdm(total=len(chapter_files), unit="chapter", desc="Assembling Chapters")
    
    for chapter_idx, chapter_file in enumerate(chapter_files):
        chapter_path = os.path.join(temp_audio_dir, chapter_file)
        chapter_name_display = chapter_file[:30] + "..." if len(chapter_file) > 30 else chapter_file
        
        with open(chapter_path, "wb") as chapter_output:
            # Create progress bar for lines in this chapter
            line_assembly_bar = tqdm(
                total=len(chapter_line_map[chapter_file]), 
                unit="line", 
                desc=f"Chapter {chapter_idx+1}/{len(chapter_files)}: {chapter_name_display}"
            )
            
            # Process each line in the correct order
            for line_index in sorted(chapter_line_map[chapter_file]):
                line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.aac")
                
                # Read the line audio and append it to the chapter file
                with open(line_audio_path, "rb") as line_input:
                    chapter_output.write(line_input.read())
                
                line_assembly_bar.update(1)
            
            line_assembly_bar.close()
        
        chapter_assembly_bar.update(1)
        yield f"Assembled chapter {chapter_idx+1}/{len(chapter_files)}: {chapter_name_display}"
    
    chapter_assembly_bar.close()
    yield "Completed assembling all chapters"
    
    # Post-processing steps
    post_processing_bar = tqdm(total=len(chapter_files)*2, unit="task", desc="Post Processing")
    
    # Add silence to each chapter file
    for chapter_idx, chapter in enumerate(chapter_files):
        add_silence_to_audio_file_by_appending_pre_generated_silence(temp_audio_dir, chapter)
        post_processing_bar.update(1)
        yield f"Added silence to chapter {chapter_idx+1}/{len(chapter_files)}"

    m4a_chapter_files = []

    # Convert all chapter files to M4A format
    for chapter_idx, chapter in enumerate(chapter_files):
        chapter_name = chapter.split('.')[0]
        m4a_chapter_files.append(f"{chapter_name}.m4a")
        # Convert to M4A as raw AAC have problems with timestamps and metadata
        convert_audio_file_formats("aac", "m4a", temp_audio_dir, chapter_name)
        post_processing_bar.update(1)
        yield f"Converted chapter {chapter_idx+1}/{len(chapter_files)} to M4A"
    
    post_processing_bar.close()
    
    # Clean up temp line audio files
    yield "Cleaning up temporary files"
    shutil.rmtree(temp_line_audio_dir)
    yield "Temporary files cleanup complete"

    if generate_m4b_audiobook_file:
        # Merge all chapter files into a final m4b audiobook
        yield "Creating M4B audiobook file..."
        merge_chapters_to_m4b(book_path, m4a_chapter_files)
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files)
        convert_audio_file_formats("m4a", output_format, "generated_audiobooks", "audiobook")
        yield f"Audiobook in {output_format} format created successfully"

async def process_audiobook_generation(voice_option, narrator_gender, output_format, book_path):
    is_kokoro_api_up, message = await check_if_kokoro_api_is_up(async_openai_client)

    if not is_kokoro_api_up:
        raise Exception(message)

    generate_m4b_audiobook_file = False

    if output_format == "M4B (Chapters & Cover)":
        generate_m4b_audiobook_file = True

    if voice_option == "Single Voice":
        yield "\n🎧 Generating audiobook with a **single voice**..."
        await asyncio.sleep(1)
        async for line in generate_audio_with_single_voice(output_format.lower(), narrator_gender, generate_m4b_audiobook_file, book_path):
            yield line
    elif voice_option == "Multi-Voice":
        yield "\n🎭 Generating audiobook with **multiple voices**..."
        await asyncio.sleep(1)
        async for line in generate_audio_with_multiple_voices(output_format.lower(), narrator_gender, generate_m4b_audiobook_file, book_path):
            yield line

    yield f"\n🎧 Audiobook is generated ! You can now download it in the Download section below. Click on the blue download link next to the file name."

async def main():
    os.makedirs("generated_audiobooks", exist_ok=True)

    # Default values
    book_path = "./sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.epub"
    generate_m4b_audiobook_file = False
    output_format = "aac"

    # Prompt user for voice selection
    print("\n🎙️ **Audiobook Voice Selection**")
    voice_option = input("🔹 Enter **1** for **Single Voice** or **2** for **Multiple Voices**: ").strip()

    # Prompt user for audiobook type selection
    print("\n🎙️ **Audiobook Type Selection**")
    print("🔹 Do you want the audiobook in M4B format (the standard format for audiobooks) with chapter timestamps and embedded book cover ? (Needs calibre and ffmpeg installed)")
    print("🔹 OR do you want a standard audio file in either of ['aac', 'm4a', 'mp3', 'wav', 'opus', 'flac', 'pcm'] formats without any of the above features ?")
    audiobook_type_option = input("🔹 Enter **1** for **M4B audiobook format** or **2** for **Standard Audio File**: ").strip()

    if audiobook_type_option == "1":
        is_calibre_installed = check_if_calibre_is_installed()

        if not is_calibre_installed:
            print("⚠️ Calibre is not installed. Please install it first and make sure **calibre** and **ebook-meta** commands are available in your PATH.")
            return
        
        is_ffmpeg_installed = check_if_ffmpeg_is_installed()

        if not is_ffmpeg_installed:
            print("⚠️ FFMpeg is not installed. Please install it first and make sure **ffmpeg** and **ffprobe** commands are available in your PATH.")
            return

        # Check if a path is provided via command-line arguments
        if len(sys.argv) > 1:
            book_path = sys.argv[1]
            print(f"📂 Using book file from command-line argument: **{book_path}**")
        else:
            # Ask user for book file path if not provided
            input_path = input("\n📖 Enter the **path to the book file**, needed for metadata and cover extraction. (Press Enter to use default): ").strip()
            if input_path:
                book_path = input_path
            print(f"📂 Using book file: **{book_path}**")

        print("✅ Book path set. Proceeding...\n")

        generate_m4b_audiobook_file = True
    else:
        # Prompt user for audio format selection
        print("\n🎙️ **Audiobook Output Format Selection**")
        output_format = input("🔹 Choose between ['aac', 'm4a', 'mp3', 'wav', 'opus', 'flac', 'pcm']. ").strip()

        if(output_format not in ["aac", "m4a", "mp3", "wav", "opus", "flac", "pcm"]):
            print("\n⚠️ Invalid output format! Please choose from the give options")
            return
        
    # Prompt user for narrator's gender selection
    print("\n🎙️ **Audiobook Narrator Voice Selection**")
    narrator_gender = input("🔹 Enter **male** if you want the book to be read in a male voice or **female** if you want the book to be read in a female voice: ").strip()

    if narrator_gender not in ["male", "female"]:
        print("\n⚠️ Invalid narrator gender! Please choose from the give options")
        return

    start_time = time.time()

    if voice_option == "1":
        print("\n🎧 Generating audiobook with a **single voice**...")
        async for line in generate_audio_with_single_voice(output_format, narrator_gender, generate_m4b_audiobook_file, book_path):
            print(line)
    elif voice_option == "2":
        print("\n🎭 Generating audiobook with **multiple voices**...")
        async for line in generate_audio_with_multiple_voices(output_format, narrator_gender, generate_m4b_audiobook_file, book_path):
            print(line)
    else:
        print("\n⚠️ Invalid option! Please restart and enter either **1** or **2**.")
        return

    print(f"\n🎧 Audiobook is generated ! The audiobook is saved as **audiobook.{"m4b" if generate_m4b_audiobook_file else output_format}** in the **generated_audiobooks** directory in the current folder.")

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"\n⏱️ **Execution Time:** {execution_time:.6f} seconds\n✅ Audiobook generation complete!")

if __name__ == "__main__":
    asyncio.run(main())