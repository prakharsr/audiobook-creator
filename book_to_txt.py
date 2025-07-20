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

import re
import sys
import time
import textract
import traceback
import shlex
import os
import subprocess
from utils.run_shell_commands import check_if_calibre_is_installed, run_shell_command_secure, validate_file_path_allowlist

def validate_book_path(book_path):
    """
    Validates that a book file path is safe using allowlist approach.
    
    Args:
        book_path (str): The path to the book file
        
    Returns:
        bool: True if path is safe and file exists, False otherwise
    """
    if not book_path or not isinstance(book_path, str):
        return False
    
    # Use allowlist-based validation  
    if not validate_file_path_allowlist(book_path):
        return False
        
    # Check if file exists and is readable
    try:
        return os.path.exists(book_path) and os.access(book_path, os.R_OK)
    except (OSError, TypeError):
        return False

def extract_text_from_book_using_textract(book_path):
    # Validate book path first
    if not validate_book_path(book_path):
        raise ValueError(f"Invalid or unsafe book path: {book_path}")
        
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
    # Validate book path first
    if not validate_book_path(book_path):
        raise ValueError(f"Invalid or unsafe book path: {book_path}")
        
    # Get ebook-convert binary path securely
    allowed_commands = ['which']
    ebook_convert_bin_result = run_shell_command_secure("which ebook-convert", allowed_commands)
    
    if not ebook_convert_bin_result or not ebook_convert_bin_result.stdout.strip():
        raise RuntimeError("ebook-convert command not found")
        
    ebook_convert_bin_path = ebook_convert_bin_result.stdout.strip()

    # Build secure command as list
    command = [ebook_convert_bin_path, book_path, "extracted_book.txt"]
    
    # Execute the command securely using our centralized function
    allowed_ebook_commands = ['ebook-convert']
    result = run_shell_command_secure(command, allowed_ebook_commands)
    
    if not result or result.returncode != 0:
        error_msg = result.stderr if result else "Unknown error"
        raise RuntimeError(f"Failed to convert book: {error_msg}")

    # Open the resulting text file and read its contents
    try:
        with open("extracted_book.txt", "r", encoding='utf-8') as f:
            book_text = f.read()
    except (OSError, IOError) as e:
        raise RuntimeError(f"Failed to read extracted book: {e}")

    return book_text

def fix_unterminated_quotes(text: str):
    """
    Fixes genuinely unterminated quotes in dialogue.
    
    This function is conservative and only fixes quotes when there's a clear
    imbalance in quote counts within dialogue paragraphs. It does NOT add
    quotes at the beginning of lines arbitrarily.
    
    Args:
        text (str): The input text to process
        
    Returns:
        str: Text with fixed quotes
    """
    lines = text.splitlines()
    fixed_lines = []
    
    for line in lines:
        if not line.strip():
            # Empty line, add as-is
            fixed_lines.append(line)
            continue
            
        # Count quotes in this line
        quote_count = line.count('"')
        
        # Only process lines that likely contain dialogue and have odd number of quotes
        if quote_count > 0 and quote_count % 2 == 1:
            # Check if this looks like dialogue (starts with quote or has attribution)
            stripped = line.strip()
            
            # Case 1: Line starts with quote but doesn't end with one
            # Example: "Hello there, how are you?
            if stripped.startswith('"') and not stripped.endswith('"'):
                # Check if this is likely incomplete dialogue
                # Look for sentence endings that should have closing quotes
                if any(stripped.endswith(punct) for punct in ['.', '!', '?', ',']):
                    line = line + '"'
                    
            # Case 2: Line has dialogue attribution but missing opening quote
            # Example: She said, Hello there!"
            elif stripped.endswith('"') and not stripped.startswith('"'):
                # Look for common dialogue attribution patterns
                attribution_patterns = [
                    r'\b(said|asked|replied|answered|shouted|whispered|muttered|declared)\b.*"',
                    r'\b(he|she|they|I)\s+(said|asked|replied|answered|shouted|whispered|muttered|declared)\b.*"',
                ]
                
                for pattern in attribution_patterns:
                    if re.search(pattern, stripped, re.IGNORECASE):
                        # Find where the actual quote content starts (after attribution)
                        # Look for the attribution and add quote after it
                        match = re.search(r'\b(said|asked|replied|answered|shouted|whispered|muttered|declared)\b\s*', stripped, re.IGNORECASE)
                        if match:
                            insert_pos = match.end()
                            line = stripped[:insert_pos] + '"' + stripped[insert_pos:]
                        break
                        
            # Case 3: Line has quote in middle but unbalanced
            # Be very conservative here - only fix obvious cases
            elif quote_count == 1 and any(word in stripped.lower() for word in ['said', 'asked', 'replied', 'shouted', 'whispered']):
                # This might be dialogue with attribution, but be cautious
                # Only fix if we can clearly identify the pattern
                pass  # For now, leave these alone to avoid errors
                
        fixed_lines.append(line)
    
    return "\n".join(fixed_lines)

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
    text_decoding_option: str = "textract"
):
    # Early validation of book path
    if not validate_book_path(book_path):
        raise ValueError(f"Invalid or unsafe book file: {book_path}. Please check the file path and permissions.")
    
    if text_decoding_option == "calibre":
        text: str = extract_text_from_book_using_calibre(book_path)
    else:
        text: str = extract_text_from_book_using_textract(book_path)

    # Replace various Unicode characters with ASCII equivalents
    text = (text.replace("\u201c", '"')      # Left double quotation mark
                .replace("\u201d", '"')      # Right double quotation mark
                .replace("\u2018", "'")      # Left single quotation mark
                .replace("\u2019", "'")      # Right single quotation mark
                .replace("\u2014", "-")      # Em dash
                .replace("\u2013", "-")      # En dash
                .replace("\u2026", "...")    # Horizontal ellipsis
                .replace("\u00a0", " ")      # Non-breaking space
                .replace("\u2010", "-")      # Hyphen
                .replace("\u2011", "-")      # Non-breaking hyphen
                .replace("\u2012", "-")      # Figure dash
                .replace("\u2015", "-")      # Horizontal bar
                .replace("\u00ab", '"')      # Left-pointing double angle quotation mark
                .replace("\u00bb", '"')      # Right-pointing double angle quotation mark
                .replace("\u2039", "'")      # Single left-pointing angle quotation mark
                .replace("\u203a", "'")      # Single right-pointing angle quotation mark
                .replace("\u201e", '"')      # Double low-9 quotation mark
                .replace("\u201a", "'")      # Single low-9 quotation mark
                .replace("\u2032", "'")      # Prime
                .replace("\u2033", '"')      # Double prime
                .replace("\ufeff", "")       # Byte order mark
                .replace("\u00ad", "")       # Soft hyphen
    )
    
    text = normalize_line_breaks(text)
    text = fix_unterminated_quotes(text)

    with open("converted_book.txt", 'w', encoding='utf-8') as fout:
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

    # Early validation of book path
    print("🔍 Validating book file...")
    if not validate_book_path(book_path):
        print(f"❌ **Book validation failed**: Invalid or inaccessible book file: {book_path}")
        print("\n💡 **Troubleshooting Tips:**")
        print("   • Ensure the book file path is correct and the file exists")
        print("   • Check that the file is readable and not corrupted")
        print("   • Verify file permissions")
        return
        
    print("✅ Book file validation successful!")
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