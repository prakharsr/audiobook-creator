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
        
        # Move punctuation from outside quotes to inside quotes for TTS compatibility
        line = _move_punctuation_inside_quotes(line)
            
        # Check if line ends with dialogue that has internal punctuation
        # This must come BEFORE general punctuation check
        if _ends_with_punctuated_dialogue(line):
            processed_lines.append(line)
            continue
            
        # Check for dialogue that lacks internal punctuation and add it inside
        if _is_unpunctuated_dialogue(line):
            line = _add_punctuation_inside_dialogue(line)
            processed_lines.append(line)
            continue
            
        # Check if line already ends with proper punctuation
        if line.endswith(('.', '!', '?', ':', ';', '…')):
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
    # Look for patterns like: "Hello!" or "What?" or "Yes." or "Oh…"
    # This covers dialogue that ends the sentence
    # Also handles nested quotes like: "He said, 'Hello.'"
    # Updated to include ellipsis and handle optional spaces between punctuation and quotes
    
    # Pattern 1: Punctuation inside quotes, optionally followed by quotes
    # Examples: "Hello!" or "What?" or "Yes." or "Oh…"
    dialogue_pattern1 = r'[.!?…]\s*[\'"]?"$'
    if re.search(dialogue_pattern1, line):
        return True
    
    # Pattern 2: Quotes followed by punctuation
    # Examples: "Is that all right". or "Tell them we don't want —".
    dialogue_pattern2 = r'[\'"][.!?…]$'
    if re.search(dialogue_pattern2, line):
        return True
        
    # Look for dialogue with attribution like: "Hello," she said.
    # This is already properly punctuated
    attribution_pattern = r'",?\s+\w+\s+(said|asked|replied|whispered|shouted|exclaimed|muttered|declared|grinned|smiled|nodded|gasped|added|stepped|pulled|pointed|stopped|noticed|flipped|closed).*[.!?]$'
    if re.search(attribution_pattern, line, re.IGNORECASE):
        return True
        
    return False


def _is_unpunctuated_dialogue(line):
    """Check if a line is dialogue that lacks internal punctuation."""
    # Look for lines that end with quotes but have no punctuation before the quotes
    # Examples: "Hello" or "What are you doing" or "Stop right there"
    
    # Pattern: line ends with quote but no punctuation before the quote
    # This catches: "Hello" but not "Hello!" or "Hello."
    unpunctuated_dialogue_pattern = r'[^.!?…]["\']$'
    if re.search(unpunctuated_dialogue_pattern, line):
        return True
        
    return False


def _add_punctuation_inside_dialogue(line):
    """
    Adds punctuation inside quotes for dialogue that lacks internal punctuation.
    
    Transforms patterns like:
    - "Hello" → "Hello."
    - "What are you doing" → "What are you doing."
    - "Stop right there" → "Stop right there."
    
    Args:
        line (str): The line of text to process
        
    Returns:
        str: The line with punctuation added inside quotes
    """
    # Add period before closing quote if no punctuation exists
    # Pattern: non-punctuation character followed by quote
    line = re.sub(r'([^.!?…])(["\'])$', r'\1.\2', line)
    
    return line


def _resolve_colon_conflicts(line):
    """
    Resolve colon conflicts for Orpheus TTS voice formatting.
    
    Orpheus uses format: <|audio|>voice_name: text content<|eot_id|>
    Colons in the text content can confuse the parser, so we simply remove all colons
    and replace them with appropriate alternatives.
    
    Args:
        line (str): The line of text to process
        
    Returns:
        str: The line with all colons removed/replaced
    """
    # Simple approach: replace all colons with appropriate alternatives
    
    # Handle time references first (preserve as periods)
    # "3:30 AM" -> "3.30 AM"
    time_pattern = r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b'
    line = re.sub(time_pattern, r'\1.\2 \3', line).strip()
    
    # Handle ratios or scores 
    # "5:3" -> "5 to 3"
    ratio_pattern = r'\b(\d+):(\d+)\b'
    line = re.sub(ratio_pattern, r'\1 to \2', line)
    
    # Replace all remaining colons with dashes
    # This handles all other cases: dialogue attribution, lists, explanations, etc.
    line = line.replace(':', ' -')
    
    # Clean up any double spaces that might have been created
    line = re.sub(r'\s+', ' ', line).strip()
    
    return line


def _move_punctuation_inside_quotes(line):
    """
    Moves punctuation from outside quotes to inside quotes for TTS compatibility.
    
    Transforms patterns like:
    - "Hello". → "Hello."
    - "What"? → "What?"
    - "Stop"! → "Stop!"
    - "Wait"… → "Wait…"
    
    Args:
        line (str): The line of text to process
        
    Returns:
        str: The line with punctuation moved inside quotes
    """
    # Move punctuation from outside quotes to inside quotes
    # Pattern: quote followed by punctuation
    line = re.sub(r'(["\'])([.!?…])', r'\2\1', line)
    
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