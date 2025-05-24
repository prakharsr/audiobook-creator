import os
import asyncio
import traceback
from tqdm.asyncio import tqdm_asyncio # Use tqdm's async version for better updates
from openai import AsyncOpenAI
from dotenv import load_dotenv
from utils.llm_utils import check_if_have_to_include_no_think_token

load_dotenv()

OPENAI_BASE_URL=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL_NAME=os.environ.get("OPENAI_MODEL_NAME", "qwen3-14b")
NO_THINK_MODE = os.environ.get("NO_THINK_MODE", "true")

openai_llm_client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
model_name = OPENAI_MODEL_NAME

MAX_CONCURRENT_TASKS = 1 # Example: Increased concurrency

# Consider adding @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def enhance_text_with_emotions(text_segment):
    """Process a text segment, adding emotion tags."""
    if not text_segment.strip():
        return text_segment # Return empty/whitespace segments as is

    no_think_token = check_if_have_to_include_no_think_token()

    system_prompt = """{no_think_token}
You are an expert editor specializing in preparing book scripts for Text-to-Speech (TTS) narration. Your task is to analyze text segments and insert specific emotion tags *only* where they are strongly implied and will enhance the audio experience. You must *never* alter the original text in any other way.

**Available Emotion Tags:**
- <giggle>
- <laugh>
- <chuckle>
- <sigh>
- <groan>
- <yawn>
- <gasp>
- <sniffle>

**Strict Rules - Follow These Exactly:**

1.  **DO NOT MODIFY THE ORIGINAL TEXT:** This is the most important rule. Do not add, remove, or change *any* words, punctuation, or formatting from the original text, except for inserting the allowed emotion tags.
2.  **ONLY USE THE PROVIDED TAGS:** Do not use any tags other than the 8 listed above.
3.  **INSERT TAGS STRATEGICALLY:**
    * **Dialogue:** Place the tag *just before* the dialogue line if the emotion applies to the whole line, or *just before* the specific word/phrase within the dialogue if it's a specific sound.
    * **Attribution/Narration:** If a word like "laughed," "sighed," "gasped," etc., is present, place the tag *just before* that word. Example: "He <laugh> laughed loudly."
    * **Implied Actions:** Only add tags to narration if it *explicitly describes* one of these sounds (e.g., "A <gasp> caught in her throat."). Do *not* add tags based on general emotional descriptions.
4.  **DO NOT OVERUSE TAGS:** Only add a tag if the emotion is *clearly* stated or *very strongly* implied. If in doubt, *do not* add a tag. Less is more.
5.  **MAINTAIN SPACING:** When inserting a tag, ensure there is a single space before it and a single space after it, *unless* it is adjacent to punctuation (like quotes or periods), in which case, place it logically. Example: "<gasp> 'No!' she cried." or "He sighed <sigh>."
6.  **HANDLE QUOTES:** If adding a tag within dialogue, place it *inside* the quotation marks.
7.  **OUTPUT FORMAT:** Return *only* the text segment with the added tags. Do not add any explanations, apologies, or introductory/concluding remarks. If no tags are needed, return the *exact original text*.

**Examples:**

* **Input:** "I can't believe you did that!" she laughed.
* **Good Output:** "I can't believe you did that!" she <laugh> laughed.

* **Input:** He looked tired. "I need some sleep."
* **Good Output:** He looked tired. "<yawn> I need some sleep."

* **Input:** "Oh, dear," she said with a sigh.
* **Good Output:** "Oh, dear," she said with a <sigh> sigh.

* **Input:** The wind howled outside.
* **Good Output:** The wind howled outside. (No tag needed)

* **Input:** "Get out!" he shouted.
* **Bad Output:** "<groan> 'Get out!' he shouted." (Groan not implied)

* **Input:** "It's funny," he chuckled.
* **Bad Output:** "It's <chuckle> funny." (Tag should be by 'chuckled')
* **Good Output:** "It's funny," he <chuckle> chuckled.

* **Input:** "Stop!" she cried, gasping for air.
* **Good Output:** "Stop!" she cried, <gasp> gasping for air.

Now, analyze the following text segment and apply these rules precisely.
""".format(no_think_token=no_think_token)

    user_prompt = f"""Please analyze this text and add appropriate emotion tags:

<text_segment>
{text_segment}
</text_segment>

Return *only* the modified text segment with any needed emotion tags."""

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

        # **Basic Validation (Optional but Recommended):**
        # You could add a check here to see if the LLM radically changed the text.
        # This is complex, but a simple check might compare word counts or
        # use a diff library. If it fails, return the original text_segment.
        # For now, we trust the LLM (guided by the strong prompt).

        return enhanced_text
    except Exception as e:
        print(f"Error querying LLM for segment: '{text_segment[:50]}...': {e}")
        traceback.print_exc()
        return text_segment # Return original on error

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

async def add_tags_to_text_chunks():
    """Processes the book text in chunks."""
    try:
        with open("converted_book.txt", "r") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: converted_book.txt not found.")
        traceback.print_exc()
        return []

    # use a line-based chunker like the helper function above
    chunks = create_chunks(text, chunk_size_lines=5)

    print(f"Processing {len(chunks)} text chunks...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []

    async def process_chunk(chunk):
        async with semaphore:
            return await enhance_text_with_emotions(chunk)

    for chunk in chunks:
        tasks.append(process_chunk(chunk))

    # Use tqdm_asyncio for progress bar with asyncio
    results = await tqdm_asyncio.gather(*tasks, unit="chunk", desc="Tag Adding Progress")

    # Reassemble the book, respecting the original chunk separation
    enhanced_text = "\n".join(results)

    try:
        with open("tag_added_lines_chunks.txt", "w") as f:
            f.write(enhanced_text)
        print(f"Adding tags completed. Output written to tag_added_lines_chunks.txt")
    except IOError as e:
        print(f"Error writing output file: {e}")
        traceback.print_exc()

    return enhanced_text

def main():
    asyncio.run(add_tags_to_text_chunks())

if __name__ == "__main__":
    main()