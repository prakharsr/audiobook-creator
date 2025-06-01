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

def preprocess_text_for_tts(text):
    """
    Preprocesses text to add punctuation where necessary to prevent TTS repetition issues.
    
    This function:
    - Adds periods to titles and chapter headings that lack ending punctuation
    - Adds periods to lines that don't end with proper punctuation
    - Preserves existing punctuation and dialogue structure
    - Handles edge cases like abbreviations and dialogue
    - Resolves colon conflicts for Orpheus TTS voice formatting
    
    Args:
        text (str): The input text to preprocess
        
    Returns:
        str: The preprocessed text with proper punctuation
    """
    lines = text.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            processed_lines.append(line)
            continue
            
        # Handle special cases where we don't want to add punctuation
        if _should_skip_punctuation(line):
            processed_lines.append(line)
            continue
            
        # Handle colon conflicts for Orpheus TTS voice formatting
        line = _resolve_colon_conflicts(line)
            
        # Check if line ends with dialogue that has internal punctuation
        # This must come BEFORE general punctuation check
        if _ends_with_punctuated_dialogue(line):
            processed_lines.append(line)
            continue
            
        # Check if line already ends with proper punctuation
        if line.endswith(('.', '!', '?', ':', ';')):
            processed_lines.append(line)
            continue
            
        # Check if this is a title or chapter heading
        if _is_title_or_heading(line, i, len(lines)):
            processed_lines.append(line + '.')
            continue
            
        # Check if line ends with a comma (might be part of a larger sentence)
        if line.endswith(','):
            processed_lines.append(line)
            continue
            
        # Default: add a period if the line doesn't end with punctuation
        processed_lines.append(line + '.')
    
    return '\n'.join(processed_lines)


def _should_skip_punctuation(line):
    """Check if we should skip adding punctuation to this line."""
    # Skip lines that are just numbers or very short
    if len(line.strip()) <= 2:
        return True
        
    # Skip lines that end with common abbreviations
    abbrev_pattern = r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Co|etc|vs|vol|no|pp)\.$'
    if re.search(abbrev_pattern, line, re.IGNORECASE):
        return True
        
    return False


def _is_title_or_heading(line, line_index, total_lines):
    """Check if this line is a title or chapter heading."""
    # First line is likely a title
    if line_index == 0:
        return True
        
    # Lines starting with "Chapter" are headings
    if re.match(r'^Chapter\s+\d+', line, re.IGNORECASE):
        return True
        
    # Lines that are short and don't contain common sentence words might be headings
    if len(line.split()) <= 6 and not any(word.lower() in line.lower() for word in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']):
        # Additional check: if the previous line was empty, this might be a heading
        if line_index > 0:
            return True
            
    return False


def _ends_with_punctuated_dialogue(line):
    """Check if line ends with dialogue that already has proper punctuation."""
    # Look for patterns like: "Hello!" or "What?" or "Yes."
    # This covers dialogue that ends the sentence
    # Also handles nested quotes like: "He said, 'Hello.'"
    dialogue_pattern = r'[.!?][\'"]?"$'
    if re.search(dialogue_pattern, line):
        return True
        
    # Look for dialogue with attribution like: "Hello," she said.
    # This is already properly punctuated
    attribution_pattern = r'",?\s+\w+\s+(said|asked|replied|whispered|shouted|exclaimed|muttered|declared|grinned|smiled|nodded|gasped|added|stepped|pulled|pointed|stopped|noticed|flipped|closed).*[.!?]$'
    if re.search(attribution_pattern, line, re.IGNORECASE):
        return True
        
    return False


def _resolve_colon_conflicts(line):
    """
    Resolve colon conflicts for Orpheus TTS voice formatting.
    
    Orpheus uses format: <|audio|>voice_name: text content<|eot_id|>
    Colons in the text content can confuse the parser, so we replace them
    with alternative punctuation in common cases.
    
    Args:
        line (str): The line of text to process
        
    Returns:
        str: The line with colon conflicts resolved
    """
    # Handle chapter headings (most common case)
    # "Chapter 1: The Beginning" -> "Chapter 1 - The Beginning"
    chapter_pattern = r'^(Chapter|Part|PART)\s+(\d+|\w+):\s*(.+)$'
    match = re.match(chapter_pattern, line, re.IGNORECASE)
    if match:
        prefix, number, title = match.groups()
        return f"{prefix} {number} - {title}"
    
    # Handle time references (e.g., "3:30 AM" -> "3.30 AM")
    time_pattern = r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b'
    line = re.sub(time_pattern, r'\1.\2 \3', line).strip()
    
    # Handle ratios or scores (e.g., "5:3" -> "5 to 3")
    ratio_pattern = r'\b(\d+):(\d+)\b'
    line = re.sub(ratio_pattern, r'\1 to \2', line)
    
    # Handle titles with colons, but preserve dialogue attribution and lists
    # Check if it's dialogue attribution (character name followed by dialogue)
    dialogue_attribution_pattern = r'^[A-Z][a-zA-Z\s]+\s+(said|asked|replied|whispered|shouted|exclaimed|muttered|declared|grinned|smiled|nodded|gasped|added|stepped|pulled|pointed|stopped|noticed|flipped|closed):\s*["\']'
    if re.search(dialogue_attribution_pattern, line, re.IGNORECASE):
        return line  # Preserve dialogue attribution format
    
    # Check if it's a list format (e.g., "He had three things: item1, item2, item3")
    list_pattern = r':\s*[a-zA-Z].*,.*\.'
    if re.search(list_pattern, line):
        return line  # Preserve list format
    
    # Handle title-style colons (e.g., "Book Title: Subtitle" -> "Book Title - Subtitle")
    title_colon_pattern = r'^([^:]{1,40}):\s*(.+)$'
    match = re.match(title_colon_pattern, line)
    if match and len(match.group(1).split()) <= 6:  # Likely a title if first part is short
        prefix, suffix = match.groups()
        # Only replace if it looks like a title, not a sentence
        if not re.search(r'\b(said|asked|replied|whispered|shouted|exclaimed|had|was|were|have|has|included|contained)\b', prefix, re.IGNORECASE):
            line = f"{prefix} - {suffix}"
    
    return line


def preprocess_text_segments_for_tts(text_segments):
    """
    Preprocesses a list of text segments for TTS.
    
    Args:
        text_segments (list): List of text segments to preprocess
        
    Returns:
        list: List of preprocessed text segments
    """
    return [preprocess_text_for_tts(segment) for segment in text_segments]