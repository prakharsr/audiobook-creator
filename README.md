# Audiobook Creator

## Overview

Audiobook Creator is an open-source project designed to convert books in various text formats (e.g., EPUB, PDF, etc.) into fully voiced audiobooks with intelligent character voice attribution. It leverages modern Natural Language Processing (NLP), Large Language Models (LLMs), and Text-to-Speech (TTS) technologies to create an engaging and dynamic audiobook experience. The project features professional-grade TTS engines including [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) and [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) with advanced async parallel processing, offering exceptional audio quality with memory-efficient processing. The project is licensed under the GNU General Public License v3.0 (GPL-3.0), ensuring that it remains free and open for everyone to use, modify, and distribute.

Sample multi voice audio for a short story generated using *Orpheus TTS* (with added emotions and better sounding audio) : https://audio.com/prakhar-sharma/audio/sample-orpheus-multi-voice-audiobook-orpheus

Sample multi voice audio for a short story generated using *Kokoro TTS* : https://audio.com/prakhar-sharma/audio/generated-sample-multi-voice-audiobook-kokoro

Watch the demo video:

[![Watch the demo video](https://img.youtube.com/vi/E5lUQoBjquo/maxresdefault.jpg)](https://www.youtube.com/watch?v=E5lUQoBjquo)

<details>
<summary>The project consists of three main components:</summary>

1. **Text Cleaning and Formatting (`book_to_txt.py`)**:
   - Extracts and cleans text from a book file (e.g., `book.epub`).
   - Normalizes special characters, fixes line breaks, and corrects formatting issues such as unterminated quotes or incomplete lines.
   - Extracts the main content between specified markers (e.g., "PROLOGUE" and "ABOUT THE AUTHOR").
   - Outputs the cleaned text to `converted_book.txt`.

2. **Character Identification and Metadata Generation (`identify_characters_and_output_book_to_jsonl.py`)**:
   - Identifies characters in the text using Named Entity Recognition (NER) with the GLiNER model.
   - Assigns gender and age scores to characters using an LLM via an OpenAI-compatible API.
   - Outputs two files:
     - `speaker_attributed_book.jsonl`: Each line of text annotated with the identified speaker.
     - `character_gender_map.json`: Metadata about characters, including name, age, gender, and gender score.

3. **Audiobook Generation (`generate_audiobook.py`)**:
   - Converts the cleaned text (`converted_book.txt`) or speaker-attributed text (`speaker_attributed_book.jsonl`) into an audiobook using advanced TTS models.
   - **Multi-Engine Support**: Compatible with both Kokoro TTS ([Hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)) and Orpheus TTS ([Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)) with engine-specific voice mapping.
   - Offers two narration modes:
     - **Single-Voice**: Uses a single voice for narration and another voice for dialogues for the entire book.
     - **Multi-Voice**: Assigns different voices to characters based on their gender scores.
   - Saves the audiobook in the selected output format to `generated_audiobooks/audiobook.{output_format}`.
</details>

## Key Features

- **Advanced TTS Engine Support**: Seamlessly switch between Kokoro and Orpheus TTS engines via environment configuration
- **Async Parallel Processing**: Optimized for concurrent request handling with significant performance improvements and faster audiobook generation.
- **Gradio UI App**: Create audiobooks easily with an easy to use, intuitive UI made with Gradio.
- **M4B Audiobook Creation**: Creates compatible audiobooks with covers, metadata, chapter timestamps etc. in M4B format.
- **Multi-Format Input Support**: Converts books from various formats (EPUB, PDF, etc.) into plain text.
- **Multi-Format Output Support**: Supports various output formats: AAC, M4A, MP3, WAV, OPUS, FLAC, PCM, M4B.
- **Docker Support**: Use pre-built docker images/ build using docker compose to save time and for a smooth user experience. 
- **Emotion Tags Addition**: Emotion tags which are supported in Orpheus TTS can be added to the book's text intelligently using an LLM to enhance character voice expression.
- **Character Identification**: Identifies characters and infers their attributes (gender, age) using advanced NLP techniques and LLMs.
- **Customizable Audiobook Narration**: Supports single-voice or multi-voice narration with narrator gender preference for enhanced listening experiences.
- **Progress Tracking**: Includes progress bars and execution time measurements for efficient monitoring.
- **Open Source**: Licensed under GPL v3.

## Sample Text and Audio

<details>
<summary>Expand</summary>

- `sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.epub`: A sample short story in epub format as a starting point.
- `sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.pdf`: A sample short story in pdf format as a starting point.
- `sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.txt`: A sample short story in txt format as a starting point.
- `sample_book_and_audio/converted_book.txt`: The cleaned output after text processing.
- `sample_book_and_audio/speaker_attributed_book.jsonl`: The generated speaker-attributed JSONL file.
- `sample_book_and_audio/character_gender_map.json`: The generated character metadata.
- `sample_book_and_audio/sample_orpheus_multi_voice_audiobook.m4b`: The generated sample multi-voice audiobook in M4B format with cover and chapters from the story and with added emotion tags, generated using Orpheus TTS.
- `sample_book_and_audio/sample_orpheus_multi_voice_audiobook.mp3`: The generated sample multi-voice MP3 audio file from the story, with added emotion tags, generated using Orpheus TTS.
- `sample_book_and_audio/sample_kokoro_multi_voice_audiobook.m4b`: The generated sample multi-voice audiobook in M4B format with cover and chapters from the story, generated using Kokoro TTS.
- `sample_book_and_audio/sample_kokoro_multi_voice_audio.mp3`: The generated sample multi-voice MP3 audio file from the story, generated using Kokoro TTS.
- `sample_book_and_audio/sample_kokoro_single_voice_audio.mp3`: The generated sample single-voice MP3 audio file from the story, generated using Kokoro TTS.
</details>

## Get Started

### Initial Setup
- Install [Docker](https://www.docker.com/products/docker-desktop/)
- Make sure host networking is enabled in your docker setup : https://docs.docker.com/engine/network/drivers/host/. Host networking is currently supported in Linux and in docker desktop. To use with [docker desktop, follow these steps](https://docs.docker.com/engine/network/drivers/host/#docker-desktop)
- Set up your LLM and expose an OpenAI-compatible endpoint (e.g., using LM Studio with `qwen3-14b`).
- **Set up TTS Engine**: Choose between Kokoro or Orpheus TTS models:

   ### Option 1: Kokoro TTS (Recommended for most users)
   Set up the Kokoro TTS model via [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI). To get started, run the docker image using the following command:

   For CUDA based GPU inference (Apple Silicon GPUs currently not supported, use CPU based inference instead). Choose the value of TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE based on [this guide](https://github.com/prakharsr/audiobook-creator/?tab=readme-ov-file#parallel-batch-inferencing-of-audio-for-faster-audio-generation)

   ```bash
  docker run \
    --name kokoro_service \
    --restart always \
    --network host \
    --gpus all \
    ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.1 \
    uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug \
    --workers {TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE}
   ```

   For CPU based inference. In this case you can keep number of workers as 1 as only mostly GPU based inferencing benefits from parallel workers and batch requests.

   ```bash
  docker run \
    --name kokoro_service \
    --restart always \
    --network host \
    ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.1 \
    uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug \
    --workers 1
   ```

   ### Option 2: Orpheus TTS (High-Quality and More Expressive Audio and Support for Emotion Tags)
   Experience premium audio quality with the Orpheus TTS FastAPI server featuring vLLM backend and bfloat16/ float16/ float32 precision. Set up Orpheus TTS using my dedicated [Orpheus TTS FastAPI](https://github.com/prakharsr/Orpheus-TTS-FastAPI) repository.

   **IMPORTANT:** Choose only highest possible precision (bf16/ fp16/ fp32) and this vLLM based FastAPI server as I noticed that using quantized versions of Orpheus or even using float16 GGUF with llama.cpp gave me audio quality issues and artifacts (repeated lines in audio/ extended audio with no spoken text but weird noises/ audio hallucinations/ some other issues). The linked repository for FastAPI server also has some additional improvements to fix such issues by detecting decoding errors and having a retry mechanism which tries to fix these audio issues automatically. 

   **Setup Instructions:**
   Please follow the complete setup instructions from the [Orpheus TTS FastAPI repository](https://github.com/prakharsr/Orpheus-TTS-FastAPI) as it contains all the necessary configuration details, installation steps, and optimization guides.

- Create a .env file from .env_sample and configure it with the correct values. Make sure you follow the instructions mentioned at the top of .env_sample to avoid errors.
   ```bash
   cp .env_sample .env
   ```
- After this, choose between the below options for the next step to run the audiobook creator app: 

   <details>
   <summary>Quickest Start (docker run)</summary>

   - Make sure your .env is configured correctly and your LLM and TTS engine (Kokoro/Orpheus) are running. In the same folder where .env is present, run the below command
   - Choose between the types of inference:
   
      For CUDA based GPU inference (Apple Silicon GPUs currently not supported, use CPU based inference instead)

      ```bash
      docker run \
         --name audiobook_creator \
         --restart always \
         --network host \
         --gpus all \
         --env-file .env \
         -v model_cache:/app/model_cache \
         ghcr.io/prakharsr/audiobook_creator_gpu:v1.4
      ```

      For CPU based inference

      ```bash
      docker run \
         --name audiobook_creator \
         --restart always \
         --network host \
         --env-file .env \
         -v model_cache:/app/model_cache \
         ghcr.io/prakharsr/audiobook_creator_cpu:v1.4
      ```
   - Wait for the models to download and then navigate to http://localhost:7860 for the Gradio UI
   </details>

   <details>
   <summary>Quick Start (docker compose)</summary>

   - Clone the repository
      ```bash 
      git clone https://github.com/prakharsr/audiobook-creator.git

      cd audiobook-creator
      ```
   - Make sure your .env is configured correctly and your LLM is running
   - If TTS docker container Kokoro is already running, you can either stop and remove it or comment the kokoro_fastapi service and depends_on param in docker compose. If its not running then it will automatically start when you run docker compose up command
   - For Orpheus TTS, please refer to the [Orpheus TTS FastAPI repository](https://github.com/prakharsr/Orpheus-TTS-FastAPI) for setup instructions as it provides the optimal implementation.
   - Copy the .env file into the audiobook-creator folder
   - Choose between the types of inference:
   
      For CUDA based GPU inference (Apple Silicon GPUs currently not supported, use CPU based inference instead). Choose the value of TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE based on [this guide](https://github.com/prakharsr/audiobook-creator/?tab=readme-ov-file#parallel-batch-inferencing-of-audio-for-faster-audio-generation) and set the value in kokoro_fastapi service and env variable.

      ```bash
      cd docker/gpu

      docker compose up --build
      ```

      For CPU based inference. In this case you can keep number of workers as 1 as only mostly GPU based inferencing benefits from parallel workers and batch requests.

      ```bash
      cd docker/cpu

      docker compose up --build
      ```
   - Wait for the models to download and then navigate to http://localhost:7860 for the Gradio UI
   </details>

   <details>
   <summary>Direct run (via uv)</summary>

   1. Clone the repository
      ```bash 
      git clone https://github.com/prakharsr/audiobook-creator.git

      cd audiobook-creator
      ```
   2. Make sure your .env is configured correctly and your LLM and TTS engine (Kokoro/Orpheus) are running
   3. Copy the .env file into the audiobook-creator folder
   4. Install uv 
      ```bash
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```
   5. Create a virtual environment with Python 3.12:
      ```bash
      uv venv --python 3.12
      ```
   5. Activate the virtual environment:
      ```bash
      source .venv/bin/activate
      ```
   6. Install Pip 24.0:
      ```bash
      uv pip install pip==24.0
      ```
   7. Install dependencies (choose CPU or GPU version):
      ```bash
      uv pip install -r requirements_cpu.txt
      ```
      ```bash
      uv pip install -r requirements_gpu.txt
      ```
   8. Upgrade version of six to avoid errors:
      ```bash
      uv pip install --upgrade six==1.17.0
      ```
   9. Install [calibre](https://calibre-ebook.com/download) (Optional dependency, needed if you need better text decoding capabilities, wider compatibility and want to create M4B audiobook). Also make sure that calibre is present in your PATH. For MacOS, do the following to add it to the PATH:
      ```bash
      deactivate
      echo 'export PATH="/Applications/calibre.app/Contents/MacOS:$PATH"' >> .venv/bin/activate
      source .venv/bin/activate
      ```
   10. Install [ffmpeg](https://www.ffmpeg.org/download.html) (Needed for audio output format conversion and if you want to create M4B audiobook)
   11. In the activated virtual environment, run `uvicorn app:app --host 0.0.0.0 --port 7860` to run the Gradio app. After the app has started, navigate to `http://127.0.0.1:7860` in the browser.
   </details>

### Parallel batch inferencing of audio for faster audio generation

- Choose the value of **TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE** based on your available VRAM to accelerate the generation of audio by using parallel batch inferencing. This variable defines the max number of parallel requests that can be made to TTS FastAPI for faster audio generation.

#### For Kokoro TTS:
- This variable is used while setting up the number of workers in Kokoro docker container and as an env variable, so make sure you set the same values for both of them. 
- You can consider setting this value to your available (VRAM/ 2) and play around with the value to see if it works best. 
- If you are unsure then a good starting point for this value can be a value of 2. 
- If you face issues of running out of memory then consider lowering the value for both workers and for the env variable.

#### For Orpheus TTS:
- The Orpheus TTS FastAPI server with vLLM backend provides native async parallel processing capabilities with significant performance improvements (up to 4x faster for long texts).
- The `TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE` parameter controls the number of concurrent requests to the Orpheus server, taking advantage of its advanced async processing pipeline.
- **Recommended values**: Start with 4-8 for most setups. Higher values (8-16) can be used with sufficient VRAM and GPU processing power.
- The server automatically handles intelligent text chunking and parallel token generation, providing optimal performance without manual tuning.
- For specific configuration and optimization guidance, refer to the [Orpheus TTS FastAPI repository](https://github.com/prakharsr/Orpheus-TTS-FastAPI) documentation.

## Roadmap

Planned future enhancements:

-  ⏳ Add support for choosing between various languages which are currently supported by Kokoro and Orpheus.
-  ✅ Process text and add emotion tags to the text in Orpheus TTS to enhance character voice expression
-  ✅ Add support for [Orpheus](https://github.com/canopyai/Orpheus-TTS). Orpheus supports 
emotion tags for a more immersive listening experience.
-  ✅ Support batch inference for TTS engines to speed up audiobook generation
-  ✅ Give choice to the user to select the voice in which they want the book to be read (male voice/ female voice)
-  ✅ Add support for running the app through docker.
-  ✅ Create UI using Gradio.
-  ✅ Try different voice combinations using `generate_audio_samples.py` and update the voice mappings to use better voices. 
-  ✅ Add support for the these output formats: AAC, M4A, MP3, WAV, OPUS, FLAC, PCM, M4B.
-  ✅ Add support for using calibre to extract the text and metadata for better formatting and wider compatibility.
-  ✅ Add artwork and chapters, and convert audiobooks to M4B format for better compatibility.
-  ✅ Give option to the user for selecting the audio generation format.
-  ✅ Add extended pause when chapters end once chapter recognition is in place.
-  ✅ Improve single-voice narration with a different dialogue voice from the narrator's voice.
-  ✅ Read out only the dialogue in a different voice instead of the entire line in that voice.

## Support

For issues or questions, open an issue on the [GitHub repository](https://github.com/prakharsr/audiobook-creator/issues).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or pull request to fix a bug or add features.

## Donations

If you find this project useful and would like to support my work, consider donating:  
[PayPal](https://paypal.me/prakharsr)

---

Enjoy creating audiobooks with this project! If you find it helpful, consider giving it a ⭐ on GitHub.
