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
from utils.file_utils import read_json

def load_voice_mappings():
    """Load the voice mappings from the JSON file."""
    return read_json("static_files/voice_map.json")

def get_narrator_and_dialogue_voices(engine_name: str, narrator_gender: str):
    """
    Get narrator and dialogue voices for single-voice mode.
    
    Args:
        engine_name (str): TTS engine name ("kokoro" or "orpheus")
        narrator_gender (str): Gender of narrator ("male" or "female")
    
    Returns:
        tuple: (narrator_voice, dialogue_voice)
    """
    voice_mappings = load_voice_mappings()
    
    if engine_name not in voice_mappings:
        raise ValueError(f"Engine '{engine_name}' not found in voice mappings")
    
    engine_voices = voice_mappings[engine_name]
    
    if narrator_gender == "male":
        narrator_voice = engine_voices["male_narrator"]
        dialogue_voice = engine_voices["male_dialogue"]
    else:  # female
        narrator_voice = engine_voices["female_narrator"]
        dialogue_voice = engine_voices["female_dialogue"]
    
    return narrator_voice, dialogue_voice

def get_voice_for_character_score(engine_name: str, narrator_gender: str, character_gender_score: int):
    """
    Get voice for a character based on narrator gender preference and character's gender score for multi-voice mode.
    
    The narrator_gender determines which score map to use (male_score_map vs female_score_map),
    while the character_gender_score determines which voice within that map.
    
    Args:
        engine_name (str): TTS engine name ("kokoro" or "orpheus")
        narrator_gender (str): User's narrator gender preference ("male" or "female")
        character_gender_score (int): Character's gender score (0-10)
    
    Returns:
        str: Voice identifier for the character
    """
    voice_mappings = load_voice_mappings()
    
    if engine_name not in voice_mappings:
        raise ValueError(f"Engine '{engine_name}' not found in voice mappings")
    
    engine_voices = voice_mappings[engine_name]
    
    # Select the appropriate score map based on NARRATOR gender preference
    if narrator_gender == "male":
        score_map = engine_voices["male_score_map"]
    else:  # female
        score_map = engine_voices["female_score_map"]
    
    # Convert score to string for JSON key lookup
    score_key = str(character_gender_score)
    
    if score_key in score_map:
        return score_map[score_key]
    else:
        # Fallback to narrator voice (score 0) if character score not found
        return score_map["0"]

def get_narrator_voice_for_character(engine_name: str, narrator_gender: str):
    """
    Get narrator voice based on user's narrator gender preference.
    
    Args:
        engine_name (str): TTS engine name ("kokoro" or "orpheus")
        narrator_gender (str): User's narrator gender preference ("male" or "female")
    
    Returns:
        str: Voice identifier for the narrator (score 0 from the appropriate score map)
    """
    voice_mappings = load_voice_mappings()
    
    if engine_name not in voice_mappings:
        raise ValueError(f"Engine '{engine_name}' not found in voice mappings")
    
    engine_voices = voice_mappings[engine_name]
    
    # Select the appropriate score map based on narrator gender preference
    if narrator_gender == "male":
        score_map = engine_voices["male_score_map"]
    else:  # female
        score_map = engine_voices["female_score_map"]
    
    # Return the narrator voice (score 0)
    return score_map["0"] 