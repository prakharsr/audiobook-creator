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

import traceback


async def check_tts_api(client, model, voice):
    FORMAT = "wav" if model == "orpheus" else "aac"
    try:
        async with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            response_format=FORMAT,
            speed=0.85,
            input="Hello, how are you ?",
        ) as response:
            return True, None
    except Exception as e:
        traceback.print_exc()
        return (
            False,
            "The TTS API is not working. Please check if the .env file is correctly set up and the TTS API is up. Error: "
            + str(e),
        )
