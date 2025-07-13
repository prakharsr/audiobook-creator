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
import traceback
from dotenv import load_dotenv

load_dotenv()

TTS_MODEL = os.environ.get("TTS_MODEL", "kokoro")

async def check_if_audio_generator_api_is_up(client):
    try:
        voice = None
        if TTS_MODEL == "kokoro":
            voice = "af_heart"
        elif TTS_MODEL == "orpheus":
            voice = "tara"
        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=voice,
            response_format="wav",  # Changed to WAV for consistency
            speed=0.85,
            input="Hello, how are you ?",
            timeout=3600
        ) as response:
            return True, None
    except Exception as e:
        traceback.print_exc()
        return False, f"The {TTS_MODEL.upper()} API is not working. Please check if the .env file is correctly set up and the {TTS_MODEL.upper()} API is up. Error: " + str(e)