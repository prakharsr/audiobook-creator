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
import os
from dotenv import load_dotenv
import random
import asyncio
from openai import AsyncOpenAI

load_dotenv()

NO_THINK_MODE = os.environ.get("NO_THINK_MODE", "true")

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # Base delay in seconds
MAX_DELAY = 60.0  # Maximum delay in seconds

def check_if_have_to_include_no_think_token():
    if NO_THINK_MODE == True or NO_THINK_MODE == "true":
        return "/no_think"
    else:
        return ""

async def check_if_llm_is_up(async_openai_client, model_name):
    try:
        response = await async_openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": f"{check_if_have_to_include_no_think_token()} Hello, this is a health test. Reply with any word if you're working."}
            ]
        )
        
        return True, response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return False, "Your configured LLM is not working. Please check if the .env file is correctly set up. Error: " + str(e)

async def generate_audio_with_retry(client: AsyncOpenAI, tts_model: str, text_to_speak: str, voice_to_speak_in: str, max_retries=MAX_RETRIES):
    """
    Generate audio with retry mechanism and exponential backoff.
    
    Args:
        client: The AsyncOpenAI client instance
        tts_model: The TTS model to use
        text_to_speak: The text to convert to speech
        voice_to_speak_in: The voice to use for TTS
        max_retries: Maximum number of retry attempts
        
    Returns:
        bytearray: Audio data buffer
        
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Create an in-memory buffer for the audio data
            audio_buffer = bytearray()
            
            # Generate audio for the part
            async with client.audio.speech.with_streaming_response.create(
                model=tts_model,
                voice=voice_to_speak_in,
                response_format="wav",
                speed=0.85,
                input=text_to_speak,
                timeout=3600
            ) as response:
                async for chunk in response.iter_bytes():
                    audio_buffer.extend(chunk)
            
            # If we reach here, the request was successful
            if attempt > 0:
                print(f"Successfully generated audio after {attempt} retry attempts")
            
            return audio_buffer
            
        except Exception as e:
            last_exception = e
            
            # Check if this is a connection-related error
            connection_errors = [
                "connection",
                "timeout",
                "network",
                "request",
                "http",
                "server",
                "api",
                "rate",
                "limit"
            ]
            
            error_message = str(e).lower()
            is_connection_error = any(error_keyword in error_message for error_keyword in connection_errors)
            
            if attempt < max_retries and is_connection_error:
                # Calculate delay with exponential backoff and jitter
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0, 0.1) * delay
                total_delay = delay + jitter
                
                print(f"Connection error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                print(f"Retrying in {total_delay:.2f} seconds...")
                
                await asyncio.sleep(total_delay)
                continue
            else:
                # Either max retries reached or non-connection error
                print(f"Failed to generate audio after {attempt + 1} attempts: {e}")
                break
    
    # If we reach here, all retry attempts failed
    raise Exception(f"Failed to generate audio after {max_retries + 1} attempts. Last error: {last_exception}")