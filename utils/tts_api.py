import asyncio
import random

from openai import AsyncOpenAI

from config.constants import API_KEY, BASE_URL


async_openai_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

async def generate_tts_with_retry(
    model, voice, text, response_format, speed=0.85, max_retries=5, task_id=None
):
    """
    Generate TTS audio with retry logic and exponential backoff.

    Args:
        client: AsyncOpenAI client
        model: TTS model to use
        voice: Voice to use
        text: Text to convert to speech
        response_format: Audio format (wav, aac, etc.)
        speed: Speech speed
        max_retries: Maximum number of retry attempts
        task_id: Task ID for cancellation checking

    Returns:
        bytearray: Audio data buffer

    Raises:
        Exception: If all retries are exhausted or task is cancelled
    """
    # Define minimum acceptable audio sizes (in bytes)
    MIN_AUDIO_SIZES = {
        "wav": 1000,  # WAV files should be at least 1KB for meaningful audio
        "aac": 500,  # AAC files can be smaller due to compression
        "mp3": 500,  # MP3 files can be smaller due to compression
        "m4a": 500,  # M4A files can be smaller due to compression
    }

    min_size = MIN_AUDIO_SIZES.get(response_format, 500)

    # Preprocess and validate text
    text = text.strip()
    if not text or len(text) < 2:
        raise Exception("Empty text provided for TTS generation")

    if not any(c.isalnum() for c in text):
        # Text contains no alphanumeric characters, might be problematic
        print(f"WARNING: Text contains no alphanumeric characters: '{text[:50]}...'")
        raise Exception("Text contains no alphanumeric characters")

    for attempt in range(max_retries + 1):
        try:

            audio_buffer = bytearray()

            async with async_openai_client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                response_format=response_format,
                speed=speed,
                input=text,
            ) as response:
                if response.status_code != 200:
                    error_msg = f"TTS API returned status {response.status_code}"
                    try:
                        error_content = await response.aread()
                        error_msg += f": {error_content.decode()}"
                    except Exception:
                        pass

                    if attempt < max_retries:
                        print(
                            f"Attempt {attempt + 1} failed with status {response.status_code}, retrying..."
                        )
                        continue
                    else:
                        raise Exception(error_msg)

                async for chunk in response.iter_bytes():
                    audio_buffer.extend(chunk)

            # Validate audio buffer size
            if len(audio_buffer) >= min_size:
                print(
                    f"✅ TTS success: Generated {len(audio_buffer)} bytes (min: {min_size})"
                )
                return audio_buffer
            else:
                error_msg = f"TTS returned insufficient audio data: {len(audio_buffer)} bytes (minimum: {min_size})"
                if attempt < max_retries:
                    print(f"Attempt {attempt + 1}: {error_msg}, retrying...")
                    # Add longer delay for insufficient data issues
                    delay = (2**attempt) + random.uniform(1, 3)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise Exception(
                        f"{error_msg} after all retries. Text: '{text[:100]}...'"
                    )

        except asyncio.CancelledError:
            # Don't retry on cancellation
            raise
        except Exception as e:
            error_msg = str(e)

            # Check if this is a retryable error
            retryable_errors = [
                "peer closed connection",
                "connection reset",
                "timeout",
                "network",
                "temporary failure",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "connection aborted",
                "connection refused",
                "connection error",
                "read timeout",
                "write timeout",
                "incomplete read",
                "broken pipe",
                "socket error",
                "http error 5",  # 5xx server errors
                "internal server error",
                "server error",
                "insufficient audio data",
            ]

            is_retryable = any(
                error_phrase in error_msg.lower() for error_phrase in retryable_errors
            )

            if attempt < max_retries and is_retryable:
                # Exponential backoff with jitter
                delay = (2**attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed with retryable error: {error_msg}")
                print(f"Retrying in {delay:.2f} seconds...")
                print(f"Text being processed: '{text[:100]}...'")
                await asyncio.sleep(delay)
                continue
            else:
                # Either max retries reached or non-retryable error
                if attempt >= max_retries:
                    print(f"All {max_retries + 1} attempts failed for TTS generation")
                    print(f"Final error: {error_msg}")
                    print(f"Text that failed: '{text[:200]}...'")
                else:
                    print(f"Non-retryable error: {error_msg}")
                raise e

    # This should never be reached, but just in case
    raise Exception("Unexpected end of retry loop")


def select_tts_voice(model, narrator_gender):
    if narrator_gender == "male":
        if model == "kokoro":
            narrator_voice = "am_puck"
        else:
            narrator_voice = "leo"
    else:
        if model == "kokoro":
            narrator_voice = "af_heart"
        else:
            narrator_voice = "tara"
    return narrator_voice
