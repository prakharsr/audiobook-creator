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

import json
import os
import asyncio
import traceback
from tqdm.asyncio import tqdm_asyncio # Use tqdm's async version for better updates
from openai import AsyncOpenAI
from dotenv import load_dotenv
from utils.llm_utils import check_if_have_to_include_no_think_token, clean_thinking_tags
import tiktoken

load_dotenv()

EMOTION_TAG_ADDITION_LLM_BASE_URL=os.environ.get("EMOTION_TAG_ADDITION_LLM_BASE_URL", "http://localhost:1234/v1")
EMOTION_TAG_ADDITION_LLM_API_KEY=os.environ.get("EMOTION_TAG_ADDITION_LLM_API_KEY", "lm-studio")
EMOTION_TAG_ADDITION_LLM_MODEL_NAME=os.environ.get("EMOTION_TAG_ADDITION_LLM_MODEL_NAME", "openai/gpt-oss-20b")
NO_THINK_MODE = os.environ.get("NO_THINK_MODE", "true")
EMOTION_TAG_ADDITION_LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE = int(os.environ.get("EMOTION_TAG_ADDITION_LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE", 1))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "500"))  # Max tokens per chunk for emotion processing
EMOTION_CONTEXT_WINDOW_SIZE = int(os.environ.get("EMOTION_CONTEXT_WINDOW_SIZE", "2"))  # Number of lines before/after emotion keyword

_TOKENIZER = None

def get_tokenizer():
    """
    Get or initialize the global tokenizer.
    Uses cl100k_base encoding (GPT-4/GPT-3.5) as a good approximation for most LLMs.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo, etc.)
            # Provides good approximation for most modern LLMs
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
            print("✅ Initialized tiktoken (cl100k_base) for token counting")
            print(f"📊 Token Configuration:")
            print(f"   Max Input Tokens per Chunk: {MAX_INPUT_TOKENS} tokens")
        except Exception as e:
            print(f"⚠️  Failed to initialize tokenizer: {e}")
    return _TOKENIZER

def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken. Returns 0 if tokenizer unavailable.
    Note: Uses cl100k_base encoding as approximation - actual count may vary by model.
    """
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return 0

openai_llm_client = AsyncOpenAI(base_url=EMOTION_TAG_ADDITION_LLM_BASE_URL, api_key=EMOTION_TAG_ADDITION_LLM_API_KEY)
model_name = EMOTION_TAG_ADDITION_LLM_MODEL_NAME

def join_lines_to_paragraphs(text):
    """
    Join consecutive non-empty lines into paragraphs for LLM processing.
    Uses a special marker (|||) to preserve original line boundaries for later splitting.
    
    Args:
        text (str): Text with each line separate
        
    Returns:
        tuple: (paragraph_text, line_structure)
            - paragraph_text: Text with lines joined, using ||| as line boundary markers
            - line_structure: List of line counts per paragraph for splitting back
    """
    lines = text.split('\n')
    paragraphs = []
    line_structure = []  # Track how many original lines each paragraph contains
    
    # Use special marker to preserve line boundaries
    LINE_MARKER = ' ||| '
    
    current_paragraph = []
    
    for line in lines:
        if line.strip():  # Non-empty line
            current_paragraph.append(line)
        else:  # Empty line - paragraph boundary
            if current_paragraph:
                # Join lines with marker to preserve boundaries
                paragraphs.append(LINE_MARKER.join(current_paragraph))
                line_structure.append(len(current_paragraph))
                current_paragraph = []
            # Keep empty line as paragraph separator
            paragraphs.append('')
            line_structure.append(0)  # 0 means empty line
    
    # Don't forget last paragraph
    if current_paragraph:
        paragraphs.append(LINE_MARKER.join(current_paragraph))
        line_structure.append(len(current_paragraph))
    
    return '\n'.join(paragraphs), line_structure

def split_paragraphs_to_lines(paragraph_text, original_line_structure):
    """
    Split processed paragraphs back to original line structure using the ||| marker.
    
    Args:
        paragraph_text (str): Text with paragraphs joined using ||| markers
        original_line_structure (list): Line counts per paragraph from join_lines_to_paragraphs
        
    Returns:
        str: Text split back to original line structure
    """
    # Use the same marker as join function
    LINE_MARKER = ' ||| '
    
    paragraphs = paragraph_text.split('\n')
    result_lines = []
    
    for i, paragraph in enumerate(paragraphs):
        if i >= len(original_line_structure):
            # Safety check - shouldn't happen
            result_lines.append(paragraph)
            continue
            
        line_count = original_line_structure[i]
        
        if line_count == 0:  # Empty line
            result_lines.append('')
        else:
            # Split on the marker to restore original lines
            split_lines = paragraph.split(LINE_MARKER)
            
            # Clean up the marker in case LLM didn't preserve it exactly
            # (sometimes LLM might add spaces or slightly modify it)
            if len(split_lines) != line_count:
                # Marker wasn't preserved perfectly - try alternative splitting
                print(f"Warning: Expected {line_count} lines but got {len(split_lines)} after splitting on marker")
                # Fall back to smart splitting based on structure
                result_lines.append(paragraph)  # Keep as single line if can't split properly
            else:
                # Perfect - marker was preserved
                result_lines.extend(split_lines)
    
    return '\n'.join(result_lines)

def fix_orphaned_tags_and_punctuation(text, original_text):
    """
    Fix orphaned emotion tags and punctuation issues while preserving line count.
    
    Args:
        text (str): Text with potential orphaned emotion tags
        original_text (str): Original text for reference
        
    Returns:
        str: Fixed text with orphaned tags relocated and punctuation corrected
    """
    import re
    
    lines = text.split('\n')
    original_lines = original_text.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        fixed_line = line
        
        # 1. Detect orphaned emotion tags (lines with mostly just a tag and minimal text)
        emotion_tags = re.findall(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>', line)
        if emotion_tags:
            # Remove tags to check remaining content
            line_without_tags = re.sub(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*', '', line).strip()
            
            # If line has only tags or very minimal content (< 10 chars), it's likely orphaned
            if len(line_without_tags) < 10:
                print(f"Detected potential orphaned tag in line: '{line}'")
                
                # Try to move tag to previous line if it looks like dialogue
                if i > 0 and i-1 < len(fixed_lines):
                    prev_line = fixed_lines[i-1]
                    # Check if previous line is dialogue (contains quotes)
                    if '"' in prev_line:
                        # Move the emotion tag to within the dialogue
                        tag = emotion_tags[0]  # Take first tag
                        fixed_prev_line = move_tag_to_dialogue(prev_line, tag)
                        if fixed_prev_line != prev_line:
                            fixed_lines[i-1] = fixed_prev_line
                            # Instead of removing line, restore original without tags
                            fixed_line = line_without_tags if line_without_tags else original_lines[i] if i < len(original_lines) else ""
                            print(f"Moved tag to previous dialogue line, preserved current line")
                        else:
                            # If couldn't move to dialogue, remove tags but keep the line
                            fixed_line = line_without_tags if line_without_tags else original_lines[i] if i < len(original_lines) else ""
                            print(f"Removed orphaned tag, preserved line content")
                    else:
                        # Remove tags but keep the line content to preserve line count
                        fixed_line = line_without_tags if line_without_tags else original_lines[i] if i < len(original_lines) else ""
                        print(f"Removed orphaned tag, preserved line content")
                else:
                    # Remove tags but keep the line content to preserve line count
                    fixed_line = line_without_tags if line_without_tags else original_lines[i] if i < len(original_lines) else ""
                    print(f"Removed orphaned tag, preserved line content")
        
        # 2. Fix punctuation issues in dialogue with emotion tags
        if '"' in fixed_line and any(tag in fixed_line for tag in ['<laugh>', '<chuckle>', '<sigh>', '<cough>', '<sniffle>', '<groan>', '<yawn>', '<gasp>']):
            fixed_line = fix_dialogue_punctuation(fixed_line)
        
        # Always append the line to preserve line count, even if it's empty
        fixed_lines.append(fixed_line)
    
    return '\n'.join(fixed_lines)

def move_tag_to_dialogue(dialogue_line, emotion_tag):
    """
    Move an emotion tag into dialogue, placing it before punctuation if present.
    
    Args:
        dialogue_line (str): Line containing dialogue
        emotion_tag (str): Emotion tag to insert (e.g., '<laugh>')
        
    Returns:
        str: Modified dialogue line with tag inserted
    """
    import re
    
    # Find dialogue content within quotes
    dialogue_pattern = r'"([^"]*)"'
    match = re.search(dialogue_pattern, dialogue_line)
    
    if match:
        dialogue_content = match.group(1)
        
        # Check if dialogue ends with punctuation
        if dialogue_content and dialogue_content[-1] in '.!?,:;':
            # Insert tag before the punctuation
            new_dialogue = dialogue_content[:-1] + f" {emotion_tag}" + dialogue_content[-1]
        else:
            # Add tag and punctuation at the end
            new_dialogue = dialogue_content + f" {emotion_tag}."
        
        # Replace the dialogue content in the original line
        new_line = dialogue_line.replace(f'"{dialogue_content}"', f'"{new_dialogue}"')
        return new_line
    
    return dialogue_line

def fix_dialogue_punctuation(line):
    """
    Fix punctuation issues in dialogue lines with emotion tags.
    
    Args:
        line (str): Line containing dialogue with emotion tags
        
    Returns:
        str: Line with fixed punctuation
    """
    import re
    
    # Pattern to find dialogue with emotion tags that might need punctuation fixes
    dialogue_pattern = r'"([^"]*<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>[^"]*)"'
    
    def fix_match(match):
        dialogue_content = match.group(1)
        
        # Check if the dialogue content ends with proper punctuation after the tag
        if not re.search(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*[.!?]', dialogue_content):
            # If tag is at the end without punctuation, add it
            if dialogue_content.endswith('>'):
                dialogue_content += '.'
        
        return f'"{dialogue_content}"'
    
    return re.sub(dialogue_pattern, fix_match, line)

def detect_remaining_orphaned_tags(text):
    """
    Detect any remaining orphaned emotion tags after attempted fixes.
    
    Args:
        text (str): Text to check for orphaned tags
        
    Returns:
        list: List of lines that still contain orphaned emotion tags
    """
    import re
    
    orphaned_lines = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        emotion_tags = re.findall(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>', line)
        if emotion_tags:
            # Remove tags to check remaining content
            line_without_tags = re.sub(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*', '', line).strip()
            
            # If line has only tags or very minimal content (< 10 chars), it's still orphaned
            # But now we don't treat this as a fatal error since we preserve line count
            if len(line_without_tags) < 10:
                print(f"Warning: Potential orphaned tag preserved at line {i+1}: '{line}'")
                # Don't add to orphaned_lines since we're preserving these lines now
    
    # Since we now preserve line count, we return empty list to indicate no fatal orphans
    return orphaned_lines

def postprocess_emotion_tags(enhanced_text, original_text):
    """
    Comprehensive postprocessing to validate and clean emotion tags output.
    
    Args:
        enhanced_text (str): Text returned by the LLM with emotion tags
        original_text (str): Original text segment before processing
        
    Returns:
        dict: {
            'text': str,           # Validated text (cleaned or original)
            'success': bool,       # True if validation passed, False if reverted
            'reverted': bool,      # True if we had to revert to original
            'reason': str or None  # Reason for reversion if failed
        }
    """
    # Define allowed emotion tags (official Orpheus TTS supported tags)
    ALLOWED_TAGS = {'<laugh>', '<chuckle>', '<sigh>', '<cough>', '<sniffle>', '<groan>', '<yawn>', '<gasp>'}
    
    try:
        # 1. Remove any backticks that the LLM might have added
        cleaned_text = enhanced_text.replace('`', '')
        
        # 2. Check for invalid emotion tags
        import re
        found_tags = set(re.findall(r'<[^>]+>', cleaned_text))
        invalid_tags = found_tags - ALLOWED_TAGS
        
        # Also check for common structural markup that might leak from prompts
        structural_tags = {'<text_segment>', '</text_segment>', '<text>', '</text>', 
                          '<input>', '</input>', '<output>', '</output>'}
        found_structural = found_tags & structural_tags
        
        if invalid_tags:
            if found_structural:
                reason = f"Structural markup leaked from prompt: {found_structural}"
            else:
                reason = f"Invalid emotion tags found: {invalid_tags}"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # 3. Check if text was significantly modified beyond adding tags
        # Remove all emotion tags to compare core content
        text_without_tags = re.sub(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>', '', cleaned_text)
        
        # Remove ALL whitespace for content comparison - we only care about actual words/characters
        def remove_all_whitespace(text):
            return re.sub(r'\s+', '', text)
        
        text_content_only = remove_all_whitespace(text_without_tags)
        original_content_only = remove_all_whitespace(original_text)
        
        if text_content_only != original_content_only:
            reason = f"Text content was modified beyond adding tags."
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # 4. Check if line breaks/newlines were preserved
        original_lines = original_text.split('\n')
        # Remove tags from enhanced text to check line structure
        enhanced_without_tags = re.sub(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*', '', cleaned_text)
        enhanced_lines = enhanced_without_tags.split('\n')
        
        if len(original_lines) != len(enhanced_lines):
            reason = f"Line structure changed ({len(original_lines)} -> {len(enhanced_lines)} lines)"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # 5. Additional validation checks
        
        # Check for malformed tags (missing < or >)
        malformed_tags = re.findall(r'<[^>]*$|^[^<]*>', cleaned_text)
        if malformed_tags:
            reason = "Malformed emotion tags found"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # Check for HTML entities that shouldn't be there
        if '&lt;' in cleaned_text or '&gt;' in cleaned_text:
            reason = "HTML entities found in emotion tags"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # Check for tags breaking words (tag in the middle of a word)
        # Look for patterns like "wo<laugh>rd"
        word_breaking_pattern = r'\w<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\w'
        if re.search(word_breaking_pattern, cleaned_text):
            reason = "Emotion tags breaking words detected"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # Check for excessive tag usage (more than 1 tag per 10 words as a safety check)
        word_count = len(original_text.split())
        tag_count = len(found_tags)
        if word_count > 0 and tag_count > max(1, word_count // 10):
            reason = f"Excessive emotion tags detected ({tag_count} tags for {word_count} words)"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # Check for duplicate consecutive tags
        consecutive_tags = re.search(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>', cleaned_text)
        if consecutive_tags:
            reason = "Consecutive emotion tags detected"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # 6. Try to fix orphaned emotion tags and punctuation issues
        fixed_text = fix_orphaned_tags_and_punctuation(cleaned_text, original_text)
        if fixed_text != cleaned_text:
            print(f"Applied orphaned tag fixes")
            cleaned_text = fixed_text
        
        # 7. Verify line count is preserved after fixing orphaned tags
        fixed_lines = cleaned_text.split('\n')
        if len(original_lines) != len(fixed_lines):
            reason = f"Line count changed after fixing orphaned tags ({len(original_lines)} -> {len(fixed_lines)} lines)"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # 8. Final check for remaining orphaned tags after fixes
        remaining_orphans = detect_remaining_orphaned_tags(cleaned_text)
        if remaining_orphans:
            reason = f"Could not fix orphaned emotion tags: {remaining_orphans}"
            print(f"Warning: {reason}. Reverting to original text.")
            return {
                'text': original_text,
                'success': False,
                'reverted': True,
                'reason': reason
            }
        
        # All validations passed
        return {
            'text': cleaned_text,
            'success': True,
            'reverted': False,
            'reason': None
        }
        
    except Exception as e:
        reason = f"Exception in postprocessing: {str(e)}"
        print(f"Error in postprocessing emotion tags: {e}")
        return {
            'text': original_text,
            'success': False,
            'reverted': True,
            'reason': reason
        }

# Consider adding @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def enhance_text_with_emotions(text_segment, semaphore=None):
    """
    Process a text segment, adding emotion tags.
    
    This function temporarily joins consecutive lines into paragraphs before LLM processing,
    then splits them back to preserve the original line structure. This ensures the LLM
    sees natural context (e.g., dialogue + attribution together) while maintaining the
    exact line-by-line format required by the audio pipeline.
    """
    if not text_segment.strip():
        return text_segment # Return empty/whitespace segments as is

    # Store original text for later restoration if needed
    original_text_segment = text_segment
    
    # Step 1: Join lines to paragraphs for better LLM context
    paragraph_text, line_structure = join_lines_to_paragraphs(text_segment)

    no_think_token = check_if_have_to_include_no_think_token()

    system_prompt = f"""{no_think_token}Reasoning: high
You are an expert editor preparing audiobook scripts for production with MULTIPLE VOICE ACTORS. Each character has their own voice actor, and there's also a separate NARRATOR voice actor who reads the narration.

**CRITICAL MENTAL MODEL - Think Like Voice Actors:**

Imagine this audiobook recording setup:
- There's a NARRATOR (one voice actor) who reads ALL narration and attribution
- There are CHARACTER voice actors (different people) who speak ONLY their character's dialogue (text in quotes)

When you see text like:
```
"Hello there!" ||| Dean laughed. ||| "How are you?"
```

The NARRATOR reads: "Dean laughed."
The CHARACTER (Dean) reads: "Hello there!" and "How are you?"

If the narration says "Dean laughed", that means:
- The NARRATOR is DESCRIBING what Dean did
- The actual LAUGH SOUND should come from DEAN's dialogue, NOT from the narrator describing it

**CRITICAL: The text may contain special line boundary markers "|||" - you MUST preserve these EXACTLY as they appear. Do NOT remove, modify, or move these markers.**

**Available Emotion Tags:**
- `<laugh>` - For laughter or when text indicates laughing
- `<chuckle>` - For light laughter or chuckling sounds
- `<sigh>` - For sighing or expressions of resignation/relief
- `<cough>` - For coughing sounds or throat clearing
- `<sniffle>` - For sniffling or nasal sounds (emotion, cold, etc.)
- `<groan>` - For groaning sounds expressing discomfort/frustration
- `<yawn>` - For yawning or expressions of tiredness
- `<gasp>` - For gasping sounds of surprise/shock

**Strict Rules - Follow These Exactly:**

1.  **DO NOT MODIFY THE ORIGINAL TEXT:** This is the most important rule. Do not add, remove, or change *any* words, punctuation, or formatting from the original text, except for inserting the allowed emotion tags.

2.  **PRESERVE ALL FORMATTING:** Maintain the exact line breaks, newlines, paragraph breaks, spacing, and indentation as they appear in the original text. Do NOT reformat, rewrap, or restructure the text in any way. Preserve the ||| markers exactly.

3.  **ONLY USE THE PROVIDED TAGS:** Do not use any tags other than the 8 listed above. Output the tags exactly as shown with angle brackets: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`.

4.  **ABSOLUTE NECESSITY - USE TAGS EXTREMELY SPARINGLY:**
    * Emotion tags should be RARE and EXCEPTIONAL - only add them when absolutely critical
    * Most text should have NO emotion tags at all
    * The text already describes emotions - tags are ONLY for when you need the actual SOUND EFFECT
    * When in doubt, DO NOT ADD A TAG - err on the side of having fewer tags

5.  **TAG PLACEMENT LOGIC (CRITICAL - WHO IS MAKING THE SOUND?):**

    **ASK YOURSELF: "Which voice actor should make this sound?"**
    
    * **If a CHARACTER should make the sound → Put tag INSIDE their dialogue quotes**
      - When narration says "X laughed", the CHARACTER X should laugh (not the narrator describing it)
      - Find X's dialogue (the text in quotes that X is speaking) and put the tag there
      - Example:
        ```
        Input:  "Would you?" ||| Dean laughed. ||| "Look at you..."
        WRONG:  "Would you?" ||| Dean laughed <laugh>. ||| "Look at you..."  ← Narrator makes sound!
        RIGHT:  "Would you? <laugh>" ||| Dean laughed. ||| "Look at you..."  ← Dean makes sound!
        ```
      
    * **NEVER place emotion tags in narration when referring to a character's action**
      - "Dean laughed" is NARRATION (spoken by narrator) - DO NOT put tag here
      - "She chuckled" is NARRATION (spoken by narrator) - DO NOT put tag here
      - "He sighed" is NARRATION (spoken by narrator) - DO NOT put tag here
      - Instead, find that character's nearby dialogue and place tag INSIDE their quotes
    
    * **Only put tags in narration if it's a pure descriptive sound without a character**
      - Example: "A laugh <laugh> echoed through the empty hall." (no specific character)
      - But if narration says "X laughed", find X's dialogue and put tag there instead

6.  **FINDING THE RIGHT DIALOGUE FOR THE TAG:**
    * Look for dialogue NEAR the narration that describes the emotion
    * Usually the character's dialogue appears just BEFORE or just AFTER the narration
    * Place the tag in that character's dialogue (inside the quotes), not in the narration
    * If you cannot find nearby dialogue from that character, DO NOT add a tag at all

7.  **MAINTAIN SPACING:** When inserting a tag, ensure there is a single space before it, *unless* it is adjacent to punctuation.

8.  **OUTPUT FORMAT:** Return *only* the text segment with the added tags. Do not add any explanations, apologies, or introductory/concluding remarks. If no tags are needed, return the *exact original text* with all formatting preserved.

**Examples:**

**CRITICAL EXAMPLES - Voice Actor Perspective:**

* **Example 1 - Character Emotion in Dialogue:**
```
Input:  "Would you?" ||| Dean Highbottom laughed. ||| "Look at you..."
```
**WRONG Output:** `"Would you?" ||| Dean Highbottom laughed <laugh>. ||| "Look at you..."`
- Why Wrong: The NARRATOR is saying "Dean Highbottom laughed" - this makes the narrator make the laugh sound!

**CORRECT Output:** `"Would you? <laugh>" ||| Dean Highbottom laughed. ||| "Look at you..."`
- Why Correct: DEAN (the character) makes the laugh sound in HIS dialogue. The narrator just describes what Dean did.
- Think: Dean is laughing WHILE speaking "Would you?" - his voice actor makes the sound.

* **Example 2 - Character Emotion with Attribution:**
```
Input:  "We won't hurt you," ||| said Sejanus. ||| Mayfair gave an ugly laugh. ||| "'Course you won't."
```
**WRONG Output:** `"We won't hurt you," ||| said Sejanus. ||| Mayfair gave an ugly laugh <laugh>. ||| "'Course you won't."`
- Why Wrong: "Mayfair gave an ugly laugh" is NARRATION - the narrator is describing Mayfair's action.

**CORRECT Output:** `"We won't hurt you," ||| said Sejanus. ||| Mayfair gave an ugly laugh. ||| "'Course you won't <laugh>."`
- Why Correct: MAYFAIR (the character) makes the laugh sound in HER dialogue "'Course you won't."
- Think: Mayfair is laughing while saying her line - her voice actor makes the sound.

* **Example 3 - When Character Has No Nearby Dialogue:**
```
Input:  The man laughed and walked away into the darkness.
```
**WRONG Output:** `The man laughed <laugh> and walked away into the darkness.`
- Why Wrong: "The man laughed" is NARRATION, but we can't find the man's dialogue to put tag in.

**CORRECT Output:** `The man laughed and walked away into the darkness.`
- Why Correct: NO TAG ADDED - the man has no dialogue here, so we cannot add the emotion sound. The narrator just describes what happened.

* **Example 4 - Inline Dialogue with Emotion:**
```
Input:  "I can't believe you did that!" she laughed.
```
**CORRECT Output:** `"I can't believe you did that! <laugh>" she laughed.`
- Why Correct: The CHARACTER is laughing while speaking her dialogue. Tag goes INSIDE her quotes.

* **Example 5 - Pure Narration (No Character):**
```
Input:  A laugh echoed through the empty hall.
```
**CORRECT Output:** `A laugh <laugh> echoed through the empty hall.`
- Why Correct: This is a pure sound description (no specific character), so narrator can make the sound.

**NO TAG NEEDED EXAMPLES:**

* **Input:** The wind howled outside.
* **Output:** The wind howled outside.
* **Why:** No human emotion sound.

* **Input:** "Get out!" he shouted.
* **Output:** "Get out!" he shouted.
* **Why:** Shouting is not in our 8 emotion tags list.

**CRITICAL REMINDERS - VOICE ACTOR MENTAL MODEL:**
- Think: "WHO is making this sound - the narrator or a character?"
- If a CHARACTER should make the sound → find their DIALOGUE and put tag INSIDE quotes
- NEVER put character emotion tags in narration (lines without quotes)
- "X laughed" in narration = narrator DESCRIBING X's action (don't tag here!)
- Find X's dialogue (text X is speaking in quotes) and tag there instead
- Use tags SPARINGLY - most text should have NO tags
- When in doubt, DO NOT add a tag
- PRESERVE the ||| markers exactly as they appear

Now, analyze the following text segment. Remember: you are preparing this for MULTIPLE VOICE ACTORS where each character has their own voice, and there's a separate narrator voice.
"""

    # Step 2: Send paragraph_text (joined) to LLM, not original line-by-line text
    user_prompt = f"""Please analyze this text and add appropriate emotion tags ONLY where ABSOLUTELY NECESSARY.

CRITICAL - THINK LIKE A VOICE ACTOR DIRECTOR:
- ASK: "Which voice actor should make this sound - the narrator or a character?"
- If narration says "X laughed" → find X's DIALOGUE and put <laugh> INSIDE X's quotes
- NEVER put character emotion tags in narration - the narrator just describes what happened
- Use tags EXTREMELY SPARINGLY - most text should have NO tags
- Only add tags for explicitly mentioned emotion sounds (laugh, chuckle, sigh, cough, sniffle, groan, yawn, gasp)
- PRESERVE the ||| markers exactly
- When in doubt, DO NOT add a tag

===== TEXT TO PROCESS =====
{paragraph_text}
===== END TEXT =====

Return *only* the modified text segment with any needed emotion tags. Do not include the delimiter lines in your response."""

    async def _call_llm():
        """Internal function to make the actual LLM call."""
        # Count tokens for monitoring
        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(user_prompt)
        total_input_tokens = system_tokens + user_tokens
        
        response = await openai_llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Set temperature
        )
        enhanced_text = response.choices[0].message.content.strip()
        cleaned_content = clean_thinking_tags(enhanced_text)
        
        # Log token usage
        output_tokens = count_tokens(cleaned_content)
        total_tokens = total_input_tokens + output_tokens
        print(f"📊 Emotion Tags - Usage: {total_input_tokens} input ({system_tokens} system + {user_tokens} user), {output_tokens} output, {total_tokens} total")

        split_result = split_paragraphs_to_lines(cleaned_content, line_structure)

        # Apply comprehensive postprocessing validation (using original line-by-line text)
        validation_result = postprocess_emotion_tags(split_result, original_text_segment)
        return validation_result
    
    try:
        # Use semaphore if provided to control concurrency
        if semaphore:
            async with semaphore:
                return await _call_llm()
        else:
            return await _call_llm()
    except Exception as e:
        print(f"Error querying LLM for segment: '{original_text_segment[:50]}...': {e}")
        traceback.print_exc()
        return {
            'text': original_text_segment,
            'success': False,
            'reverted': True,
            'reason': f"LLM query error: {str(e)}"
        }

def create_chunks(text, max_tokens=None, chunk_size_lines=5):
    """
    Splits text into chunks based on token count, trying to respect paragraphs.
    
    Args:
        text (str): Text to split into chunks
        max_tokens (int, optional): Maximum tokens per chunk. Defaults to MAX_INPUT_TOKENS.
        chunk_size_lines (int): Fallback line-based chunk size if tokenizer unavailable
        
    Returns:
        list: List of text chunks
    """
    if max_tokens is None:
        max_tokens = MAX_INPUT_TOKENS
    
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    # Initialize tokenizer
    get_tokenizer()
    
    for line in lines:
        line_tokens = count_tokens(line)
        
        # Check if adding this line would exceed token limit
        if current_chunk and (current_token_count + line_tokens > max_tokens):
            # Save current chunk and start new one
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_token_count = 0
        
        current_chunk.append(line)
        current_token_count += line_tokens
        
        # Also break on paragraph boundaries (empty lines) for natural breaks
        if not line.strip() and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_token_count = 0

    if current_chunk: # Add any remaining lines
        chunks.append("\n".join(current_chunk))
    
    # Log chunking statistics
    if chunks:
        avg_tokens = sum(count_tokens(chunk) for chunk in chunks) / len(chunks)
        print(f"📦 Created {len(chunks)} chunks | Avg tokens per chunk: {avg_tokens:.0f} | Max tokens: {max_tokens}")

    return chunks

async def process_chunk_line_by_line(chunk, semaphore=None):
    """
    Fallback function to process a chunk line by line when chunk-based processing fails.
    
    Args:
        chunk (str): The text chunk that failed chunk-based processing
        semaphore: Optional semaphore to control concurrency of LLM requests
        
    Returns:
        str: The chunk with emotion tags added line by line
    """
    lines = chunk.split('\n')
    processed_lines = []
    
    for line in lines:
        if line.strip():  # Only process non-empty lines
            try:
                # Process each line individually, passing semaphore for concurrency control
                result = await enhance_text_with_emotions(line, semaphore=semaphore)
                processed_lines.append(result['text'])
            except Exception as e:
                print(f"Error processing individual line, using original: {e}")
                processed_lines.append(line)
        else:
            # Preserve empty lines exactly
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def extract_emotion_context_windows(text, context_sentences=2):
    """
    Extract context windows around emotion-related keywords using sliding window approach.
    
    This function implements a major optimization: instead of processing the entire book
    with the LLM, we only process text segments that contain emotion-related keywords
    (laugh, chuckle, sigh, etc.). This can reduce LLM calls by 70-90% depending on the book.
    
    Algorithm:
    1. Search for emotion keywords: laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp
    2. Extract context window (N lines before/after) around each keyword match
    3. Merge overlapping windows to avoid duplicate processing
    4. Return only these focused windows for LLM processing
    5. Unprocessed text segments remain unchanged in the final output
    
    Args:
        text (str): Full text to search
        context_sentences (int): Number of lines to include before/after keyword
        
    Returns:
        tuple: (windows, original_lines)
            - windows: List of dicts with {'text': str, 'start_line': int, 'end_line': int}
            - original_lines: List of all original text lines for reassembly
    """
    import re
    
    # Emotion keywords pattern - match word boundaries to avoid partial matches
    emotion_pattern = r'\b(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)(ed|ing|s|es)?\b'
    
    lines = text.split('\n')
    windows = []
    processed_line_ranges = set()  # Track which line ranges we've already processed
    
    # Find all lines containing emotion keywords
    for line_idx, line in enumerate(lines):
        if re.search(emotion_pattern, line, re.IGNORECASE):
            # Calculate window boundaries (sentences are approximated by line groups)
            start_idx = max(0, line_idx - context_sentences)
            end_idx = min(len(lines), line_idx + context_sentences + 1)
            
            # Create a tuple representing this range
            range_key = (start_idx, end_idx)
            
            # Skip if we've already covered this range (handles overlapping windows)
            if range_key in processed_line_ranges:
                continue
            
            # Check if this range overlaps with existing windows
            overlaps = False
            for existing_start, existing_end in list(processed_line_ranges):
                # Check for overlap
                if not (end_idx <= existing_start or start_idx >= existing_end):
                    # Merge overlapping windows by extending the range
                    new_start = min(start_idx, existing_start)
                    new_end = max(end_idx, existing_end)
                    processed_line_ranges.discard((existing_start, existing_end))
                    processed_line_ranges.add((new_start, new_end))
                    overlaps = True
                    break
            
            if not overlaps:
                processed_line_ranges.add(range_key)
    
    # Convert line ranges to text windows
    for start_idx, end_idx in sorted(processed_line_ranges):
        window_lines = lines[start_idx:end_idx]
        window_text = '\n'.join(window_lines)
        windows.append({
            'text': window_text,
            'start_line': start_idx,
            'end_line': end_idx,
            'line_count': end_idx - start_idx
        })
    
    return windows, lines

async def add_tags_to_text_chunks(text_to_process):
    """Processes the book text in chunks and yields progress updates."""

    # Filter out empty lines to match JSONL processing (which skips empty lines)
    lines = text_to_process.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]  # Only keep non-empty lines
    filtered_text = '\n'.join(non_empty_lines)
    
    yield f"Filtered text to match JSONL processing: {len(lines)} -> {len(non_empty_lines)} lines"

    # OPTIMIZATION: Extract only context windows around emotion keywords
    yield "Searching for emotion-related keywords in text..."
    windows, original_lines = extract_emotion_context_windows(filtered_text, context_sentences=EMOTION_CONTEXT_WINDOW_SIZE)
    
    if not windows:
        yield "No emotion keywords found in text. Skipping LLM processing."
        # Write original text and return early
        try:
            with open("tag_added_lines_chunks.txt", "w") as f:
                f.write(filtered_text)
            yield f"No emotion tags needed. Output written to tag_added_lines_chunks.txt"
            yield f"Line count verification: Filtered={len(non_empty_lines)}, Enhanced={len(filtered_text.split('\n'))}"
        except IOError as e:
            yield f"Error writing output file: {e}"
            traceback.print_exc()
        return
    
    total_lines_to_process = sum(w['line_count'] for w in windows)
    coverage_percent = (total_lines_to_process / len(non_empty_lines)) * 100 if non_empty_lines else 0
    
    yield f"Found {len(windows)} context windows containing emotion keywords"
    yield f"Processing {total_lines_to_process} lines ({coverage_percent:.1f}% of text) instead of all {len(non_empty_lines)} lines"
    yield f"Estimated LLM call reduction: {100 - coverage_percent:.1f}%"

    # Use token-based chunker with max 4k tokens per chunk - but only on filtered windows
    chunks = []
    chunk_mappings = []  # Track which lines each chunk corresponds to
    
    for window in windows:
        window_chunks = create_chunks(window['text'], max_tokens=MAX_INPUT_TOKENS)
        for chunk in window_chunks:
            chunks.append(chunk)
            chunk_mappings.append({
                'start_line': window['start_line'],
                'end_line': window['end_line']
            })

    yield f"Processing {len(chunks)} text chunks for emotion tags (max {MAX_INPUT_TOKENS} tokens per chunk)..."

    # Create shared semaphore to control total concurrent LLM requests
    # This ensures concurrency limit is respected across both chunk and line-by-line processing
    semaphore = asyncio.Semaphore(EMOTION_TAG_ADDITION_LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE)
    progress_counter = 0
    total_chunks = len(chunks)

    async def process_chunk(chunk_index, chunk):
        nonlocal progress_counter
        # Try chunk-based processing first, passing semaphore for concurrency control
        enhancement_result = await enhance_text_with_emotions(chunk, semaphore=semaphore)
        final_result = enhancement_result['text']
        
        # Check if postprocessing failed and we need fallback
        if enhancement_result['reverted'] and enhancement_result['reason']:
            print(f"Chunk {chunk_index}: Chunk processing failed ({enhancement_result['reason']})")
            print(f"Chunk {chunk_index}: Falling back to line-by-line processing")
            try:
                # Pass semaphore to line-by-line processing to maintain concurrency control
                final_result = await process_chunk_line_by_line(chunk, semaphore=semaphore)
            except Exception as e:
                print(f"Chunk {chunk_index}: Line-by-line fallback also failed: {e}")
                final_result = chunk  # Use original chunk if both methods fail
        
        progress_counter += 1
        return {
            "index": chunk_index,
            "result": final_result,
            "progress": progress_counter
        }

    yield f"Processing {total_chunks} chunks for emotion tags (max {EMOTION_TAG_ADDITION_LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE} concurrent)..."
    
    # Create tasks with chunk indices
    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append(process_chunk(i, chunk))

    # Process tasks and yield progress updates as they complete
    results = [None] * total_chunks  # Pre-allocate to maintain order
    
    for completed_task in asyncio.as_completed(tasks):
        task_result = await completed_task
        chunk_index = task_result["index"]
        chunk_result = task_result["result"]
        current_progress = task_result["progress"]
        
        results[chunk_index] = chunk_result
        
        yield f"Processed {current_progress}/{total_chunks} emotion tag chunks..."
    
    yield f"Completed processing all {total_chunks} emotion tag chunks"

    # Reassemble the book by mapping processed windows back to original positions
    yield "Reassembling text with processed emotion tag windows..."
    
    # Start with original lines (unprocessed)
    final_lines = original_lines.copy()
    
    # Map processed chunks back to their original line positions
    for i, chunk_result in enumerate(results):
        mapping = chunk_mappings[i]
        processed_lines = chunk_result.split('\n')
        
        # Replace the corresponding lines in the final output
        start_line = mapping['start_line']
        end_line = mapping['end_line']
        
        # Calculate how many lines to replace
        lines_to_replace = end_line - start_line
        
        if len(processed_lines) == lines_to_replace:
            # Direct replacement if line count matches
            for j, processed_line in enumerate(processed_lines):
                final_lines[start_line + j] = processed_line
        else:
            # Handle line count mismatch - this shouldn't happen with our validation
            # but adding safety check
            yield f"Warning: Line count mismatch in chunk {i} (expected {lines_to_replace}, got {len(processed_lines)})"
    
    enhanced_text = '\n'.join(final_lines)
    
    # Validate that line count is preserved (using filtered text for comparison)
    enhanced_lines = enhanced_text.split('\n')
    
    if len(non_empty_lines) != len(enhanced_lines):
        error_msg = f"ERROR: Line count mismatch after emotion processing! Filtered: {len(non_empty_lines)}, Enhanced: {len(enhanced_lines)}"
        yield error_msg

        # Revert to filtered text to prevent synchronization issues
        enhanced_text = filtered_text
        yield "Reverted to filtered text to prevent synchronization issues"

    try:
        with open("tag_added_lines_chunks.txt", "w") as f:
            f.write(enhanced_text)
        yield f"Emotion tags processing completed. Output written to tag_added_lines_chunks.txt"
        enhanced_text_split = enhanced_text.split('\n')
        yield f"Line count verification: Filtered={len(non_empty_lines)}, Enhanced={len(enhanced_text_split)}"
    except IOError as e:
        yield f"Error writing output file: {e}"
        traceback.print_exc()

async def process_emotion_tags_for_jsonl_data(json_data_array):
    """
    Efficiently process emotion tags for JSONL data using optimized chunk processing.
    
    This function extracts all text from the JSONL data, processes it using the optimized
    add_tags_to_text_chunks function, then maps the enhanced text back to the original
    JSONL structure while preserving speaker attributions.
    
    Args:
        json_data_array (list): Array of JSONL objects with 'line' and 'speaker' fields
        
    Yields:
        str: Progress updates during emotion tags processing
        
    Returns:
        list: Updated JSONL array with emotion tags added to the text
    """
    # Extract all lines of text while preserving the mapping to original entries
    all_text_lines = []
    line_mappings = []  # Maps processed text lines back to original jsonl entries
    
    for i, item in enumerate(json_data_array):
        if "line" in item and item["line"] and item["line"].strip():
            all_text_lines.append(item["line"])
            line_mappings.append(i)  # Store the index in the original array
    
    if not all_text_lines:
        yield "No text found to process for emotion tags"
        yield json_data_array  # No text to process
    
    yield f"Preparing {len(all_text_lines)} lines for emotion tags processing..."
    
    # Combine all text into a single block for optimized chunk processing
    combined_text = "\n".join(all_text_lines)
    
    # Use the optimized add_tags_to_text_chunks function and yield its progress

    async for progress in add_tags_to_text_chunks(combined_text):
        yield progress

    enhanced_text = None
    with open("tag_added_lines_chunks.txt", "r") as f:
        enhanced_text = f.read()
    
    # Split the enhanced text back into individual lines
    enhanced_lines = enhanced_text.split("\n")
    
    yield f"Mapping enhanced text back to {len(line_mappings)} original entries..."
    
    # Map the enhanced lines back to the original JSONL structure
    # Handle potential line count mismatches gracefully
    for i, enhanced_line in enumerate(enhanced_lines):
        if i < len(line_mappings):
            original_index = line_mappings[i]
            json_data_array[original_index]["line"] = enhanced_line
    
    yield f"Successfully enhanced {len(line_mappings)} lines with emotion tags"
    
    # Final yield with the result
    yield json_data_array