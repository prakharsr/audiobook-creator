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
import os
import time
import json
import random
import asyncio
import traceback
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import torch
from gliner import GLiNER
import warnings
from utils.file_utils import write_jsons_to_jsonl_file, empty_file, write_json_to_file
from utils.find_book_protagonist import find_book_protagonist
from utils.llm_utils import check_if_have_to_include_no_think_token, check_if_llm_is_up
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

def download_with_progress(model_name):
    print(f"Starting download of {model_name}")
    
    # Define cache directory
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Log progress manually since tqdm might not work in Docker
    print("Download in progress - this may take several minutes...")
    
    # Download without tqdm
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_files_only=False,
        local_dir=cache_dir
    )
    
    print(f"Download complete for {model_name}")
    
    # Load the model from cache
    return GLiNER.from_pretrained(model_name, cache_dir=cache_dir)

load_dotenv()

OPENAI_BASE_URL=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL_NAME=os.environ.get("OPENAI_MODEL_NAME", "qwen3-14b")

# warnings.simplefilter("ignore")

print("\n🚀 **Downloading the GLiNER Model ...**")

async_openai_client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
model_name = OPENAI_MODEL_NAME
gliner_model = download_with_progress("urchade/gliner_large-v2.1")

print("\n🚀 **GLiNER Model Backend Selection**")

if torch.cuda.is_available():
    print("🟢 Using **CUDA** backend (NVIDIA GPU detected)")
    gliner_model = gliner_model.cuda()  # For Nvidia CUDA Accelerated GPUs
elif torch.backends.mps.is_available():
    print("🍏 Using **MPS** backend (Apple Silicon GPU detected)")
    gliner_model = gliner_model.to("mps")  # For Apple Silicon GPUs
else:
    print("⚪ Using **CPU** backend (No compatible GPU found)")

print("✅ Model is ready!\n")

def extract_dialogues(text):
    """Extract dialogue lines enclosed in quotes."""
    return re.findall(r'("[^"]+")', text)

def identify_speaker_using_named_entity_recognition(
    line_map: list[dict], 
    index: int, 
    line: str, 
    prev_speaker: str, 
    protagonist: str, 
    character_gender_map: dict
) -> str:
    """
    Identifies the speaker of a given line in a text using Named Entity Recognition (NER).

    This function analyzes the provided line and its context to determine the speaker. It uses
    a pre-trained NER model to detect entities and matches them with known characters or pronouns.
    If no entity is found, it falls back to the previous speaker or assigns a default value.

    Args:
        line_map (list[dict]): A list of dictionaries representing lines of text, where each dictionary
                              contains information about a line (e.g., the text itself).
        index (int): The index of the current line in the `line_map`.
        line (str): The current line of text to analyze.
        prev_speaker (str): The speaker identified in the previous line.
        protagonist (str): The name of the protagonist, used to resolve first-person references.
        character_gender_map (dict): A dictionary mapping character names to their genders, used to
                                     resolve third-person references.

    Returns:
        str: The identified speaker, normalized to lowercase.
    """

    current_line = line
    text = f"{current_line}"
    speaker: str = "narrator"  # Default speaker is the narrator

    # Labels for the NER model to detect
    labels = ["character", "person"]

    # Lists of pronouns for different person and gender references
    first_person_person_single_references = ["i", "me", "my", "mine", "myself"]  # First person singular
    first_person_person_collective_references = ["we", "us", "our", "ours", "ourselves"]  # First person collective
    second_person_person_references = ["you", "your", "yours", "yourself", "yourselves"]  # Second person
    third_person_male_references = ["he", "him", "his", "himself"]  # Third person male
    third_person_female_references = ["she", "her", "hers", "herself"]  # Third person female
    third_person_others_references = [
        "they", "them", "their", "theirs", "themself", "themselves", "it", "its", "itself"
    ]  # Third person neutral/unknown

    # Extract character names based on gender from the character_gender_map
    gender_scores = list(character_gender_map["scores"].values())
    male_characters = [x["name"] for x in gender_scores if x["gender"] == "male"]
    female_characters = [x["name"] for x in gender_scores if x["gender"] == "female"]
    other_characters = [x["name"] for x in gender_scores if x["gender"] == "unknown"]

    # Use the NER model to detect entities in the current line
    entities = gliner_model.predict_entities(text, labels)
    entity = entities[0] if len(entities) > 0 else None

    # If no entity is found, check previous lines (up to 5 lines back) for context
    loop_index = index - 1
    while (entity is None) and loop_index >= max(0, index - 5):
        prev_lines = "\n".join(x["line"] for x in line_map[loop_index: index])
        text = f"{prev_lines}\n{current_line}"
        entities = gliner_model.predict_entities(text, labels)
        entity = entities[0] if len(entities) > 0 else None
        loop_index -= 1

    # Determine the speaker based on the detected entity or fallback logic
    if entity is None:
        # If no entity is found, use the previous speaker or mark as unknown
        if prev_speaker == "narrator":
            speaker = "unknown"
        else:
            speaker = prev_speaker
    elif entity["text"].lower() in first_person_person_single_references:
        # First-person singular pronouns refer to the protagonist
        speaker = protagonist
    elif entity["text"].lower() in first_person_person_collective_references:
        # First-person collective pronouns refer to the previous speaker
        speaker = prev_speaker
    elif entity["text"].lower() in second_person_person_references:
        # Second-person pronouns refer to the previous speaker
        speaker = prev_speaker
    elif entity["text"].lower() in third_person_male_references:
        # Third-person male pronouns refer to the last mentioned male character
        last_male_character = male_characters[-1] if len(male_characters) > 0 else "unknown"
        speaker = last_male_character
    elif entity["text"].lower() in third_person_female_references:
        # Third-person female pronouns refer to the last mentioned female character
        last_female_character = female_characters[-1] if len(female_characters) > 0 else "unknown"
        speaker = last_female_character
    elif entity["text"].lower() in third_person_others_references:
        # Third-person neutral/unknown pronouns refer to the last mentioned neutral/unknown character
        last_other_character = other_characters[-1] if len(other_characters) > 0 else "unknown"
        speaker = last_other_character
    else:
        # If the entity is not a pronoun, use the entity text as the speaker
        speaker = entity["text"]

    return speaker.lower()

async def identify_character_gender_and_age_using_llm_and_assign_score(character_name, index, lines):
    """
    Identifies a character's gender and age using a Language Model (LLM) and assigns a gender score.

    Args:
        character_name (str): The name or description of the character.
        index (int): The index of the character's dialogue in the `lines` list.
        lines (list): A list of strings representing the text lines (dialogues or descriptions).

    Returns:
        dict: A dictionary containing the character's name, inferred age, inferred gender, and gender score.
              Example: {"name": "John", "age": "adult", "gender": "male", "gender_score": 2}
    """

    try:
        # Extract a window of dialogues around the character's line for context
        character_dialogues = lines[max(0, index - 2):index + 5]
        text_character_dialogues = "\n".join(character_dialogues)

        no_think_token = check_if_have_to_include_no_think_token()

        # System prompt to guide the LLM in inferring age and gender
        system_prompt = f"""
        {no_think_token}
        You are an expert in analyzing character names and inferring their gender and age based on the character's name and the text excerpt. Take into consideration the character name and the text excerpt and then assign the age and gender accordingly. 
        For a masculine character return the gender as 'male', for a feminine character return the gender as 'female' and for a character whose gender is neutral/ unknown return gender as 'unknown'. 
        For assigning the age, if the character is a child return the age as 'child', if the character is an adult return the age as 'adult' and if the character is an elderly return the age as 'elderly'.
        Return only the gender and age as the output. Dont give any explanation or doubt. 
        Give the output as a string in the following format:
        Age: <age>
        Gender: <gender>"""

        # User prompt containing the character name and dialogue context
        user_prompt = f"""
        Character Name/ Character Description: {character_name}

        Text Excerpt: {text_character_dialogues}
        """

        # Query the LLM to infer age and gender
        response = await async_openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        # Extract and clean the LLM's response
        age_and_gender = response.choices[0].message.content
        age_and_gender = age_and_gender.lower().strip()
        split_text = age_and_gender.split("\n")
        age_text = split_text[0]
        gender_text = split_text[1]

        # Parse age and gender from the response
        age = age_text.split(":")[1].strip()
        gender = gender_text.split(":")[1].strip()

        # Default to "adult" if age is unknown or neutral
        if age not in ["child", "adult", "elderly"]:
            age = "adult"

        # Default to "unknown" if gender is unknown or neutral
        if gender not in ["male", "female", "unknown"]:
            gender = "unknown"

        # Assign a gender score based on inferred gender and age
        gender_score = 5  # Default to neutral/unknown

        if gender == "male":
            if age == "child":
                gender_score = 4  # Slightly masculine for male children
            elif age == "adult":
                gender_score = random.choice([1, 2, 3])  # Mostly to completely masculine for male adults
            elif age == "elderly":
                gender_score = random.choice([1, 2])  # Mostly to completely masculine for elderly males
        elif gender == "female":
            if age == "child":
                gender_score = 10  # Completely feminine for female children
            elif age == "adult":
                gender_score = random.choice([7, 8, 9])  # Mostly to completely feminine for female adults
            elif age == "elderly":
                gender_score = random.choice([6, 7])  # Slightly to moderately feminine for elderly females

        # Compile character information into a dictionary
        character_info = {
            "name": character_name,
            "age": age,
            "gender": gender,
            "gender_score": gender_score
        }
        return character_info
    except Exception as e:
        print(f"Error: {e}. Defaulting to 'adult' age and 'unknown' gender in response.")
        traceback.print_exc()
        character_info = {
            "name": character_name,
            "age": "adult",
            "gender": "unknown",
            "gender_score": 5
        }
        return character_info

async def identify_characters_and_output_book_to_jsonl(text: str, protagonist):
    """
    Processes a given text to identify characters, assign gender scores, and output the results to JSONL files.

    This function performs the following steps:
    1. Clears an existing JSONL file for storing speaker-attributed lines.
    2. Identifies characters in the text using Named Entity Recognition (NER).
    3. Assigns gender and age scores to characters using a Language Model (LLM).
    4. Outputs the processed text with speaker attributions to a JSONL file.
    5. Saves the character gender and age scores to a separate JSON file.

    Args:
        text (str): The input text to be processed, typically a book or script.
        protagonist: The main character of the text, used as a reference for speaker identification.

    Outputs:
        - speaker_attributed_book.jsonl: A JSONL file where each line contains a speaker and their corresponding dialogue or narration.
        - character_gender_map.json: A JSON file containing gender and age scores for each character.
    """
    # Clear the output JSONL file
    empty_file("speaker_attributed_book.jsonl")

    yield("Identifying Characters. Progress 0%")

    # Initialize a set to track known characters
    known_characters = set()

    # Define a mapping for character gender scores and initialize with the narrator
    character_gender_map = {
        "legend": {
            "1": "completely masculine",
            "2": "mostly masculine",
            "3": "moderately masculine",
            "4": "slightly masculine",
            "5": "neutral/unknown",
            "6": "slightly feminine",
            "7": "moderately feminine",
            "8": "mostly feminine",
            "9": "almost completely feminine",
            "10": "completely feminine"
        },
        "scores": {
            "narrator": {
                "name": "narrator",
                "age": "adult",
                "gender": "female", # or male based on the user's selection in audiobook generation step 
                "gender_score": 0  # Default score for the narrator
            }
        }
    }

    # Split the text into lines and extract dialogues
    lines = text.split("\n")
    dialogues = extract_dialogues(text)
    prev_speaker = "narrator"  # Track the previous speaker
    line_map: list[dict] = []  # Store speaker-attributed lines
    dialogue_last_index = 0  # Track the last processed dialogue index

    # Process each line in the text with a progress bar
    with tqdm(total=len(lines), unit="line", desc="Identifying Characters Using Named Entity Recognition and assigning gender scores using LLM : ") as overall_pbar:
        for index, line in enumerate(lines):
            try:
                # Skip empty lines
                if not line:
                    continue

                # Check if the line contains a dialogue
                dialogue = None
                for dialogue_index in range(dialogue_last_index, len(dialogues)):
                    dialogue_inner = dialogues[dialogue_index]
                    if dialogue_inner in line:
                        dialogue_last_index = dialogue_index
                        dialogue = dialogue_inner
                        break

                # If the line contains a dialogue, identify the speaker
                if dialogue:
                    speaker = identify_speaker_using_named_entity_recognition(line_map, index, line, prev_speaker, protagonist, character_gender_map)

                    # Add the speaker and line to the line map
                    line_map.append({"speaker": speaker, "line": line})

                    # If the speaker is new, assign gender and age scores using LLM
                    if speaker not in known_characters:
                        known_characters.add(speaker)
                        character_gender_map["scores"][speaker] = await identify_character_gender_and_age_using_llm_and_assign_score(speaker, index, lines)

                    prev_speaker = speaker
                else:
                    # If no dialogue, attribute the line to the narrator
                    line_map.append({"speaker": "narrator", "line": line})

                # Update the progress bar
                overall_pbar.update(1)
                yield f"Identifying Characters. Progress: {index + 1}/{len(lines)} ({(index + 1) * 100 // len(lines)}%)"

            except Exception as e:
                # Handle errors and log them
                print(f"!!! Error !!! Index: {index}, Error: ", e)
                traceback.print_exc()

    # Write the processed lines to a JSONL file
    write_jsons_to_jsonl_file(line_map, "speaker_attributed_book.jsonl")

    # Write the character gender and age scores to a JSON file
    write_json_to_file(character_gender_map, "character_gender_map.json")

    yield "Character Identification Completed. You can now move onto the next step (Audiobook generation)."

async def process_book_and_identify_characters(book_name):
    is_llm_up, message = await check_if_llm_is_up(async_openai_client, model_name)

    if not is_llm_up:
        raise Exception(message)

    yield "Finding protagonist. Please wait..."
    protagonist = await find_book_protagonist(book_name, async_openai_client, model_name)
    f = open("converted_book.txt", "r", encoding='utf-8')
    book_text = f.read()
    yield f"Found protagonist: {protagonist}"
    await asyncio.sleep(1)

    async for update in identify_characters_and_output_book_to_jsonl(book_text, protagonist):
        yield update

async def main():
    f = open("converted_book.txt", "r", encoding='utf-8')
    book_text = f.read()

    # Ask for the protagonist's name
    print("\n📖 **Character Identification Setup**")
    protagonist = input("🔹 Enter the name of the **protagonist** (Check from Wikipedia if needed): ").strip()

    # Start processing
    start_time = time.time()
    print("\n🔍 Identifying characters and processing the book...")
    async for update in identify_characters_and_output_book_to_jsonl(book_text, protagonist):
        print(update)
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"\n⏱️ **Execution Time:** {execution_time:.6f} seconds")

    # Completion message
    print("\n✅ **Character identification complete!**")
    print("🎧 Next, run the following script to generate the audiobook:")
    print("   ➜ `python generate_audiobook.py`")
    print("\n🚀 Happy audiobook creation!\n")

if __name__ == "__main__":
    asyncio.run(main())
