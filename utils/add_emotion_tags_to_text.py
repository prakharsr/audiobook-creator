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

load_dotenv()

OPENAI_BASE_URL=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL_NAME=os.environ.get("OPENAI_MODEL_NAME", "qwen3-14b")
NO_THINK_MODE = os.environ.get("NO_THINK_MODE", "true")
LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE = int(os.environ.get("LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE", 1))

openai_llm_client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
model_name = OPENAI_MODEL_NAME

def fix_orphaned_tags_and_punctuation(text, original_text):
    """
    Fix orphaned emotion tags and punctuation issues.
    
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
                if i > 0:
                    prev_line = lines[i-1]
                    # Check if previous line is dialogue (contains quotes)
                    if '"' in prev_line:
                        # Move the emotion tag to within the dialogue
                        tag = emotion_tags[0]  # Take first tag
                        fixed_prev_line = move_tag_to_dialogue(prev_line, tag)
                        if fixed_prev_line != prev_line:
                            fixed_lines[i-1] = fixed_prev_line
                            fixed_line = ""  # Remove the orphaned line
                            print(f"Moved tag to previous dialogue line")
                        else:
                            # If couldn't move to dialogue, try to merge with previous line
                            fixed_lines[i-1] = prev_line + " " + line.strip()
                            fixed_line = ""
                            print(f"Merged orphaned tag with previous line")
        
        # 2. Fix punctuation issues in dialogue with emotion tags
        if '"' in fixed_line and any(tag in fixed_line for tag in ['<laugh>', '<chuckle>', '<sigh>', '<cough>', '<sniffle>', '<groan>', '<yawn>', '<gasp>']):
            fixed_line = fix_dialogue_punctuation(fixed_line)
        
        if fixed_line:  # Only add non-empty lines
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
            if len(line_without_tags) < 10:
                orphaned_lines.append(f"Line {i+1}: '{line}'")
    
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
        
        # 7. Final check for remaining orphaned tags after fixes
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
async def enhance_text_with_emotions(text_segment):
    """Process a text segment, adding emotion tags."""
    if not text_segment.strip():
        return text_segment # Return empty/whitespace segments as is

    no_think_token = check_if_have_to_include_no_think_token()

    system_prompt = f"""{no_think_token}
You are an expert editor specializing in preparing book scripts for Text-to-Speech (TTS) narration. Your task is to analyze text segments and insert specific emotion tags *only* where they are strongly implied and will enhance the audio experience. You must *never* alter the original text in any other way.

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
2.  **PRESERVE ALL FORMATTING:** Maintain the exact line breaks, newlines, paragraph breaks, spacing, and indentation as they appear in the original text. Do NOT reformat, rewrap, or restructure the text in any way.
3.  **ONLY USE THE PROVIDED TAGS:** Do not use any tags other than the 8 listed above. Output the tags exactly as shown with angle brackets: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`.
4.  **INSERT TAGS STRATEGICALLY:**
    * **Dialogue:** ALWAYS place emotion tags WITHIN the dialogue quotes, never after them. Place tags *just before* the punctuation if the dialogue ends with punctuation, or add punctuation after the tag if none exists.
    * **Attribution/Narration:** If a word like "laughed," "chuckled," "sighed," "coughed," "sniffled," "groaned," "yawned," "gasped," etc., is present, place the tag *just after* that word for natural audio flow. This creates better narrative flow where the listener hears the context first, then the emotion sound. Example: "He laughed `<laugh>` loudly." or "She coughed `<cough>` softly."
    * **Implied Actions:** Only add tags to narration if it *explicitly describes* one of these sounds (e.g., "A `<gasp>` caught in her throat." or "He cleared his throat with a `<cough>` cough."). Do *not* add tags based on general emotional descriptions.
    * **CRITICAL:** Never create standalone lines with only emotion tags. Always attach tags to substantial text content.
5.  **DO NOT OVERUSE TAGS:** Only add a tag if the emotion is *clearly* stated or *very strongly* implied. If in doubt, *do not* add a tag. Less is more.
6.  **MAINTAIN SPACING:** When inserting a tag, ensure there is a single space before it and a single space after it, *unless* it is adjacent to punctuation (like quotes or periods), in which case, place it logically. Example: "`<gasp>` 'No!' she cried." or "He sighed `<sigh>`."
7.  **HANDLE QUOTES:** If adding a tag within dialogue, place it *inside* the quotation marks.
8.  **PRESERVE LINE STRUCTURE:** If the input has multiple lines, empty lines, or specific line breaks, maintain them exactly. Do NOT merge lines or change the line structure.
9.  **OUTPUT FORMAT:** Return *only* the text segment with the added tags. Do not add any explanations, apologies, or introductory/concluding remarks. Do not include any structural markup, delimiters, or formatting from the input prompt. If no tags are needed, return the *exact original text* with all formatting preserved.

**Examples:**

* **Input:** "I can't believe you did that!" she laughed.
* **Good Output:** "I can't believe you did that!" she laughed `<laugh>`.

* **Input:** He looked tired. "I need some sleep."
* **Good Output:** He looked tired. "I need some sleep `<yawn>`."

* **Input:** "Look what I found! It's a treasure map!"
* **Bad Output:** "Look what I found! It's a treasure map!" `<laugh>` (Creates orphaned tag)
* **Good Output:** "Look what I found! It's a treasure map `<laugh>`!" (Tag within dialogue, before punctuation)

* **Input:** "Oh, dear," she said with a sigh.
* **Good Output:** "Oh, dear," she said with a sigh `<sigh>`.

* **Input:** He cleared his throat before speaking.
* **Good Output:** He cleared `<cough>` his throat before speaking.

* **Input:** The wind howled outside.
* **Good Output:** The wind howled outside. (No tag needed)

* **Input:** Mia held up a dusty, old map she had found in her attic. "Look what I found! It's a treasure map!".
* **Bad Output:** Mia held up a dusty, old map she had found in her attic. "Look what I found! It's a treasure map!" `<exclamation>` (Invalid tag - not in official list)
* **Good Output:** Mia held up a dusty, old map she had found in her attic. "Look what I found! It's a treasure map!" (No tag needed - excitement implied but no specific sound mentioned)

* **Input:** "Get out!" he shouted.
* **Bad Output:** "`<groan>` 'Get out!' he shouted." (Groan not implied)

* **Input:** "It's funny," he chuckled.
* **Bad Output:** "It's `<chuckle>` funny." (Tag should be by 'chuckled')
* **Good Output:** "It's funny," he `<chuckle>` chuckled.

* **Input:** "Stop!" she cried, gasping for air.
* **Good Output:** "Stop `<gasp>`!" she cried, gasping `<gasp>` for air.

* **Input:** Luna gasped in surprise.
* **Bad Output:** Luna `<gasp>` gasped in surprise. (Emotion before context - sounds unnatural)
* **Good Output:** Luna gasped `<gasp>` in surprise. (Context first, then emotion sound)

* **Input:** The five friends laughed together.
* **Bad Output:** The five friends `<laugh>` laughed together. (Emotion before context)
* **Good Output:** The five friends laughed `<laugh>` together. (Natural flow: word then sound)

* **Input:** "What are you doing here?"
* **Bad Output:** "What are you doing here?" `<sigh>` (Creates orphaned tag)
* **Good Output:** "What are you doing here `<sigh>`?" (Tag within dialogue, adds punctuation)

**Line Break Preservation Examples:**

* **Input:** 
```
"Hello there," she said.

He sighed deeply.
"I don't know what to do."
```
* **Good Output:** 
```
"Hello there," she said.

He `<sigh>` sighed deeply.
"I don't know what to do."
```

* **Input:** 
```
The room was quiet.
    "Are you okay?" she whispered.
    
    There was no response.
```
* **Good Output:** 
```
The room was quiet.
    "Are you okay?" she whispered.
    
    There was no response.
```

Now, analyze the following text segment and apply these rules precisely.
"""

    user_prompt = f"""Please analyze this text and add appropriate emotion tags:

===== TEXT TO PROCESS =====
{text_segment}
===== END TEXT =====

Return *only* the modified text segment with any needed emotion tags. Do not include the delimiter lines in your response."""

    try:
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

        # Apply comprehensive postprocessing validation
        validation_result = postprocess_emotion_tags(cleaned_content, text_segment)
        return validation_result
    except Exception as e:
        print(f"Error querying LLM for segment: '{text_segment[:50]}...': {e}")
        traceback.print_exc()
        return {
            'text': text_segment,
            'success': False,
            'reverted': True,
            'reason': f"LLM query error: {str(e)}"
        }

def create_chunks(text, chunk_size_lines=5):
    """Splits text into chunks, trying to respect paragraphs."""
    lines = text.split('\n')
    chunks = []
    current_chunk = []

    for line in lines:
        current_chunk.append(line)
        # If line is a paragraph break OR chunk size reached
        if not line.strip() or len(current_chunk) >= chunk_size_lines:
            chunks.append("\n".join(current_chunk))
            current_chunk = []

    if current_chunk: # Add any remaining lines
        chunks.append("\n".join(current_chunk))

    return chunks

async def process_chunk_line_by_line(chunk):
    """
    Fallback function to process a chunk line by line when chunk-based processing fails.
    
    Args:
        chunk (str): The text chunk that failed chunk-based processing
        
    Returns:
        str: The chunk with emotion tags added line by line
    """
    lines = chunk.split('\n')
    processed_lines = []
    
    for line in lines:
        if line.strip():  # Only process non-empty lines
            try:
                # Process each line individually
                result = await enhance_text_with_emotions(line)
                processed_lines.append(result['text'])
            except Exception as e:
                print(f"Error processing individual line, using original: {e}")
                processed_lines.append(line)
        else:
            # Preserve empty lines exactly
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

async def add_tags_to_text_chunks(text_to_process):
    """Processes the book text in chunks and yields progress updates."""

    # use a line-based chunker like the helper function above
    chunks = create_chunks(text_to_process, chunk_size_lines=5)

    yield f"Processing {len(chunks)} text chunks for emotion tags..."

    semaphore = asyncio.Semaphore(LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE)
    progress_counter = 0
    total_chunks = len(chunks)

    async def process_chunk(chunk_index, chunk):
        nonlocal progress_counter
        async with semaphore:
            # Try chunk-based processing first
            enhancement_result = await enhance_text_with_emotions(chunk)
            final_result = enhancement_result['text']
            
            # Check if postprocessing failed and we need fallback
            if enhancement_result['reverted'] and enhancement_result['reason']:
                print(f"Chunk {chunk_index}: Chunk processing failed ({enhancement_result['reason']})")
                print(f"Chunk {chunk_index}: Falling back to line-by-line processing")
                try:
                    final_result = await process_chunk_line_by_line(chunk)
                except Exception as e:
                    print(f"Chunk {chunk_index}: Line-by-line fallback also failed: {e}")
                    final_result = chunk  # Use original chunk if both methods fail
            
            progress_counter += 1
            return {
                "index": chunk_index,
                "result": final_result,
                "progress": progress_counter
            }

    yield f"Processing {total_chunks} chunks for emotion tags (max {LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE} concurrent)..."
    
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

    # Reassemble the book, respecting the original chunk separation
    enhanced_text = "\n".join(results)

    try:
        with open("tag_added_lines_chunks.txt", "w") as f:
            f.write(enhanced_text)
        yield f"Emotion tags processing completed. Output written to tag_added_lines_chunks.txt"
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