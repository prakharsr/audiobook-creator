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

import os
import re
import sys
import time
import textract
import traceback
from config.constants import TEMP_DIR
from utils.run_shell_commands import run_shell_command, check_if_calibre_is_installed

def extract_text_from_book_using_textract(book_path):
    text: str = textract.process(book_path, encoding='utf-8').decode() # decode using textract

    return text

def extract_text_from_book_using_calibre(book_path):
    """
    Extracts text from a book using Calibre's ebook-convert utility.

    Args:
        book_path (str): The path to the book file.

    Returns:
        str: The extracted text from the book.
    """
    ebook_convert_bin_result = run_shell_command("which ebook-convert")
    ebook_convert_bin_path = ebook_convert_bin_result.stdout.strip()

    # Command to convert the book into a plain text file using ebook-convert
    command = f"{ebook_convert_bin_path} '{book_path}' extracted_book.txt"
    
    # Execute the command without using a virtual environment
    result = run_shell_command(command)

    # Open the resulting text file and read its contents
    with open("extracted_book.txt", "r", encoding='utf-8') as f:
        book_text = f.read()

    return book_text

def fix_unterminated_quotes(text: str):
    """Fixes unterminated quotes in the given text."""
    lines = text.splitlines()  # Split the text into lines
    fixed_text = []
    in_quote = False  # Track if we're inside a quote

    for line in lines:
        quote_indices = [i for i, char in enumerate(line) if char == '"']  # Find all quote positions
        if not quote_indices:
            fixed_text.append(line)  # No quotes in this line, add as is
            continue

        # Process quotes in the line
        new_line = ""
        for i, char in enumerate(line):
            if char == '"':
                in_quote = not in_quote  # Toggle quote state
            new_line += char

        # If the line ends with an unterminated quote, add a closing quote
        if in_quote and not line.endswith('"'):
            new_line += '"'
            in_quote = False  # Reset quote state

        # If the line ends with an unterminated quote, add a closing quote
        if in_quote and not new_line.startswith('"'):
            new_line = '"' + new_line
            in_quote = False  # Reset quote state

        fixed_text.append(new_line)

    return "\n".join(fixed_text)  # Join lines back into a single string

def extract_main_content(text, start_marker="PROLOGUE", end_marker="ABOUT THE AUTHOR"):
    """
    Extracts the main content of a book between two markers (case-insensitive).
    Handles edge cases such as multiple marker occurrences and proper content boundaries.
    
    Args:
        text (str): The full text of the book.
        start_marker (str): The marker indicating the start of the main content.
        end_marker (str): The marker indicating the end of the main content.
    
    Returns:
        str: The extracted main content.
        
    Raises:
        ValueError: If markers are not found or if their positions are invalid.
    """
    
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        if not start_marker or not end_marker:
            raise ValueError("Markers must be non-empty strings")
        
        # Find all occurrences of markers
        start_positions = []
        end_positions = []
        pos = 0
        
        # Find all start marker positions
        while True:
            pos = text.find(start_marker, pos)
            if pos == -1:
                break
            start_positions.append(pos)
            pos += 1
            
        # Find all end marker positions
        pos = 0
        while True:
            pos = text.find(end_marker, pos)
            if pos == -1:
                break
            end_positions.append(pos)
            pos += 1
            
        # Validate marker existence
        if not start_positions:
            raise ValueError(f"Start marker '{start_marker}' not found in the text")
        if not end_positions:
            raise ValueError(f"End marker '{end_marker}' not found in the text")
            
        # Find the correct pair of markers
        start_index = start_positions[len(start_positions)-1]
        end_index = end_positions[len(end_positions)-1]
    
        if start_index is None or end_index is None:
            raise ValueError("Could not find valid marker positions with substantial content between them")
            
        # Extract and clean the content
        main_content = text[start_index:end_index].strip()
        
        # Validate extracted content
        if len(main_content) < 100:  # Adjust this threshold as needed
            raise ValueError("Extracted content is suspiciously short")
            
        # Remove any leading/trailing chapter markers or section headers
        lines = main_content.split('\n')
        while lines and (
            any(marker.lower() in lines[0].lower() 
                for marker in [start_marker, end_marker, 'chapter', 'part', 'book'])):
            lines.pop(0)
        while lines and (
            any(marker.lower() in lines[-1].lower() 
                for marker in [start_marker, end_marker, 'chapter', 'part', 'book'])):
            lines.pop()
            
        return '\n'.join(lines).strip()
    except Exception as e:
        print(e)
        traceback.print_exc()
        print("Error", e, ", not extracting main content.")
        return text

def normalize_line_breaks(text):
    # Split the text into lines
    lines = text.splitlines()
    
    # Filter out empty lines and strip any leading/trailing whitespace
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Join the lines with a single line break
    normalized_text = '\n'.join(non_empty_lines)
    
    return normalized_text

def save_book(edited_text):
    with open("converted_book.txt", "w", encoding="utf-8") as fout:
        fout.write(edited_text)
    return "📖 Book saved successfully as 'converted_book.txt'! You can now proceed to the next optional step (Identifying Characters) or move onto Audiobook generation"


def process_book_and_extract_text(
    book_path: str,
    text_decoding_option: str = "textract",
    title: str = "converted_book.txt",
):
    print(f"🔄 Processing book: {book_path}")
    if text_decoding_option == "calibre":
        text: str = extract_text_from_book_using_calibre(book_path)
    else:
        text: str = extract_text_from_book_using_textract(book_path)

    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )
    text = normalize_line_breaks(text)
    text = fix_unterminated_quotes(text)

    # Create the temp directory and book-specific subdirectory if they don't exist
    book_temp_dir = os.path.join(TEMP_DIR, title)
    os.makedirs(book_temp_dir, exist_ok=True)

    # Save the text file inside the book directory
    text_file_path = os.path.join(book_temp_dir, "converted_book.txt")
    print(f"🔄 Saving book to {text_file_path}")

    with open(text_file_path, "w", encoding="utf-8") as fout:
        fout.write(text)
        yield text


def main():
    # Default book path
    book_path = "./sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.epub"

    # Check if a path is provided via command-line arguments
    if len(sys.argv) > 1:
        book_path = sys.argv[1]
        print(f"📂 Using book file from command-line argument: **{book_path}**")
    else:
        # Ask user for book file path if not provided
        input_path = input("\n📖 Enter the **path to the book file** (Press Enter to use default): ").strip()
        if input_path:
            book_path = input_path
        print(f"📂 Using book file: **{book_path}**")

    print("✅ Book path set. Proceeding...\n")

    print("\n🔧 Text Decoding Options:\n")

    text_decoding_option = input(
        "❓ Do you want to extract and decode the text using textract or calibre ?\n"
        "📌 Use calibre for better formatted results, wider compatibility for ebook formats and if you have it installed.\n"
        "➡️ Answer (textract/calibre). Default is **textract**: "
    ).strip().lower()

    print("✍️ Decoding the book...\n")

    if(text_decoding_option == "calibre"):
        is_calibre_installed = check_if_calibre_is_installed()

        if is_calibre_installed:
            print("✅ Calibre is installed. Using calibre to decode the book...\n")
            text: str = extract_text_from_book_using_calibre(book_path)
        else:
            print("⚠️ Calibre is not installed. Please install it first and make sure **calibre** and **ebook-convert** commands are available in your PATH.")
            return
    else:
        print("✅ Using textract to decode the book...\n")
        text: str = extract_text_from_book_using_textract(book_path)

    print("✍️ Normalizing the text by replacing curly quotes and apostrophes with standard ASCII equivalents...\n")

    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").replace("\u2018", "'") # Normalize text by replacing curly quotes and apostrophes with standard ASCII equivalents

    print("✍️ Removing multiple line breaks...\n")

    text = normalize_line_breaks(text) # Remove multiple line breaks, normalize it 

    # Fix missing opening/closing quotes in dialogue
    print("\n✍️ Fixing unterminated quotes in dialogue...\n")
    text = fix_unterminated_quotes(text)

    # Ask user if they want to extract main content
    have_to_extract_main_content = input(
        "❓ Do you want to extract the **main content** of the book? (Optional)\n"
        "📌 You can also do this step manually for finer control over the audiobook text.\n"
        "➡️ Answer (yes/no). Default is **no**: "
    ).strip().lower()

    if have_to_extract_main_content == "yes":
        start_marker = input("🔹 Enter the **start marker** for the main content (case-sensitive): Default is **PROLOGUE** :").strip()
        if(not start_marker):
            start_marker = "PROLOGUE"
        end_marker = input("🔹 Enter the **end marker** for the main content (case-sensitive): Default is **ABOUT THE AUTHOR** :").strip()
        if(not end_marker):
            end_marker = "ABOUT THE AUTHOR"
        text = extract_main_content(text, start_marker=start_marker, end_marker=end_marker)
        print("✅ Main content has been extracted!\n")

    print("\n🚀 Processing complete!\n")

    with open("converted_book.txt", 'w', encoding='utf-8') as fout:
        fout.write(text)

        print("📖 Your book has been successfully cleaned and converted!")
        print("✅ Saved as: converted_book.txt (in the current working directory)\n")

        print("🔍 Please manually review the converted book and remove any unnecessary content.\n")

        print("🎭 Next Steps:")
        print("  - If you want **multiple voices**, run:")
        print("    ➜ `python identify_characters_and_output_book_to_jsonl.py`")
        print("    (This script will identify characters and assign gender & age scores.)\n")
        print("  - If you want a **single voice**, directly run:")
        print("    ➜ `python generate_audiobook.py`")
        print("    (This will generate the audiobook immediately.)\n")

        print("🚀 Happy audiobook creation!")

if __name__ == "__main__":
    main()
