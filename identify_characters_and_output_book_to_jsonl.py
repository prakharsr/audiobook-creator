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
import warnings
from utils.file_utils import empty_file
from utils.llm_utils import check_if_llm_is_up
from utils.character_extraction_llm import llm_identify_characters_and_output_to_jsonl
from dotenv import load_dotenv

load_dotenv()

CHARACTER_IDENTIFICATION_LLM_BASE_URL=os.environ.get("CHARACTER_IDENTIFICATION_LLM_BASE_URL", "http://localhost:1234/v1")
CHARACTER_IDENTIFICATION_LLM_API_KEY=os.environ.get("CHARACTER_IDENTIFICATION_LLM_API_KEY", "lm-studio")
CHARACTER_IDENTIFICATION_LLM_MODEL_NAME=os.environ.get("CHARACTER_IDENTIFICATION_LLM_MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507")

async_openai_client = AsyncOpenAI(base_url=CHARACTER_IDENTIFICATION_LLM_BASE_URL, api_key=CHARACTER_IDENTIFICATION_LLM_API_KEY)
model_name = CHARACTER_IDENTIFICATION_LLM_MODEL_NAME

def extract_dialogues(text):
    """Extract dialogue lines enclosed in quotes."""
    return re.findall(r'("[^"]+")', text)

async def identify_characters_and_output_book_to_jsonl(text: str):
    """

    TWO-PASS APPROACH: Extract all characters first, then attribute dialogue.
    
    Pass 1: Extract ALL characters from entire text (including dialogue)
    Pass 2: Attribute speakers to dialogue (pure matching, no character creation)

    Args:
        text (str): The input text to be processed, typically a book or script.

    Outputs:
        - speaker_attributed_book.jsonl: A JSONL file where each line contains a speaker and their corresponding dialogue or narration.
        - character_gender_map.json: A JSON file containing gender and age scores for each character.
    """
    # Clear the output JSONL file
    empty_file("speaker_attributed_book.jsonl")

    yield("Identifying Characters. Progress 0%")

    async for update in llm_identify_characters_and_output_to_jsonl(
        text, async_openai_client, model_name
    ):
        yield update

    yield "Character Identification Completed. You can now move onto the next step (Audiobook generation)."

async def process_book_and_identify_characters():
    is_llm_up, message = await check_if_llm_is_up(async_openai_client, model_name)

    if not is_llm_up:
        raise Exception(message)

    f = open("converted_book.txt", "r", encoding='utf-8')
    book_text = f.read()

    async for update in identify_characters_and_output_book_to_jsonl(book_text):
        yield update

async def main():
    f = open("converted_book.txt", "r", encoding='utf-8')
    book_text = f.read()

    # Start processing
    start_time = time.time()
    print("\n🔍 Identifying characters and processing the book...")
    async for update in identify_characters_and_output_book_to_jsonl(book_text):
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
