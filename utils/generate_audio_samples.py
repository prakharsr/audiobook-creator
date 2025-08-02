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

from openai import OpenAI
import itertools
import requests
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8880/v1")
TTS_API_KEY = os.environ.get("TTS_API_KEY", "not-needed")
TTS_MODEL = os.environ.get("TTS_MODEL", "kokoro")

os.makedirs("audio_samples", exist_ok=True)

client = OpenAI(
    base_url=TTS_BASE_URL, api_key=TTS_API_KEY
)

def get_available_voices():
    """Fetch available voices from the TTS API endpoint."""
    try:
        response = requests.get(f"{TTS_BASE_URL}/audio/voices")
        voices_res = response.json()
        voices = voices_res["voices"]
        # print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return []

text = """Humpty Dumpty sat on a wall.
Humpty Dumpty had a great fall.
All the king's horses and all the king's men.
Couldn't put Humpty together again."""

voices = get_available_voices()
# print("Available voices:", voices)

combinations = list(itertools.combinations(voices, 2))
all_voices_combinations = voices.copy()

for comb in combinations:
    all_voices_combinations.append("+".join(comb))

gen_for_all_combinations = input("Generate voice sample for all voice combinations ? Enter yes or no : ")
gen_for_all_combinations = gen_for_all_combinations.strip().lower()

if(gen_for_all_combinations == "yes"):
    with tqdm(total=len(all_voices_combinations), unit="line", desc="Audio Generation Progress") as overall_pbar:
        for voice in all_voices_combinations:
            with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=voice,
                response_format="wav",  # Changed to WAV for consistency
                speed=0.85,
                input=text,
                timeout=600
            ) as response:
                file_path = f"audio_samples/{voice}.wav"
                response.stream_to_file(file_path)
            overall_pbar.update(1)
else:
    with tqdm(total=len(voices), unit="line", desc="Audio Generation Progress") as overall_pbar:
        for voice in voices:
            with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=voice,
                response_format="wav",  # Changed to WAV for consistency
                speed=0.85,
                input=text,
                timeout=600
            ) as response:
                file_path = f"audio_samples/{voice}_test.wav"
                response.stream_to_file(file_path)
            overall_pbar.update(1)

print("TTS generation complete!")
