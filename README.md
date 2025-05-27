# Audiobook Creator

## Overview

Audiobook Creator is an open-source project designed to convert books in various text formats (e.g., EPUB, PDF, etc.) into fully voiced audiobooks with intelligent character voice attribution. It leverages modern Natural Language Processing (NLP), Large Language Models (LLMs), and Text-to-Speech (TTS) technologies to create an engaging and dynamic audiobook experience. The project is licensed under the GNU General Public License v3.0 (GPL-3.0), ensuring that it remains free and open for everyone to use, modify, and distribute.

Sample multi voice audio for a short story : https://audio.com/prakhar-sharma/audio/generated-sample-multi-voice-audiobook

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
   - Converts the cleaned text (`converted_book.txt`) or speaker-attributed text (`speaker_attributed_book.jsonl`) into an audiobook using your choice of TTS models:
     - **Kokoro TTS** ([Hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)) - High-quality, fast TTS with multiple voices
     - **Orpheus TTS** ([Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI)) - High-performance TTS with emotion tags, 8 voices, and OpenAI-compatible API
   - Offers two narration modes:
     - **Single-Voice**: Uses a single voice for narration and another voice for dialogues for the entire book.
     - **Multi-Voice**: Assigns different voices to characters based on their gender scores.
   - Saves the audiobook in the selected output format to `generated_audiobooks/audiobook.{output_format}`.
   </details>

## Key Features

- **Gradio UI App**: Create audiobooks easily with an easy to use, intuitive UI made with Gradio.
- **M4B Audiobook Creation**: Creates compatible audiobooks with covers, metadata, chapter timestamps etc. in M4B format.
- **Multi-Format Input Support**: Converts books from various formats (EPUB, PDF, etc.) into plain text.
- **Multi-Format Output Support**: Supports various output formats: AAC, M4A, MP3, WAV, OPUS, FLAC, PCM, M4B.
- **Docker Support**: Use pre-built docker images/ build using docker compose to save time and for a smooth user experience.
- **Text Cleaning**: Ensures the book text is well-formatted and readable.
- **Character Identification**: Identifies characters and infers their attributes (gender, age) using advanced NLP techniques.
- **Customizable Audiobook Narration**: Supports single-voice or multi-voice narration for enhanced listening experiences.
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
- `sample_book_and_audio/sample_multi_voice_audiobook.m4b`: The generated sample multi-voice audiobook in M4B format with cover and chapters from the story.
- `sample_book_and_audio/sample_multi_voice_audio.mp3`: The generated sample multi-voice MP3 audio file from the story.
- `sample_book_and_audio/sample_single_voice_audio.mp3`: The generated sample single-voice MP3 audio file from the story.
</details>

## Get Started

### Initial Setup

- Install [Docker](https://www.docker.com/products/docker-desktop/)
- Make sure host networking is enabled in your docker setup : https://docs.docker.com/engine/network/drivers/host/. Host networking is currently supported in Linux and in docker desktop. To use with [docker desktop, follow these steps](https://docs.docker.com/engine/network/drivers/host/#docker-desktop)
- Set up your LLM and expose an OpenAI-compatible endpoint (e.g., using LM Studio with `qwen3-14b`).
- **Choose and set up your TTS service** (one of the following):

  **Option A: Kokoro TTS** via [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI). To get started, run the docker image using the following command:

  For CUDA based GPU inference (Apple Silicon GPUs currently not supported, use CPU based inference instead). Choose the value of MAX_PARALLEL_REQUESTS_BATCH_SIZE based on [this guide](https://github.com/prakharsr/audiobook-creator/?tab=readme-ov-file#parallel-batch-inferencing-of-audio-for-faster-audio-generation)

  ```bash
  docker run \
   --name service \
   --restart always \
   --network host \
   --gpus all \
   ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.2 \
   uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug \
   --workers {MAX_PARALLEL_REQUESTS_BATCH_SIZE}
  ```

  For CPU based inference. In this case you can keep number of workers as 1 as only mostly GPU based inferencing benefits from parallel workers and batch requests.

  ```bash
  docker run \
   --name service \
   --restart always \
   --network host \
   ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.2 \
   uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug \
   --workers 1
  ```

  **Option B: Orpheus TTS** via [Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI). Orpheus offers high-performance TTS with emotion tags and 8 different voices. To get started:

  ```bash
  git clone https://github.com/Lex-au/Orpheus-FastAPI.git
  cd Orpheus-FastAPI

  # For GPU inference
  docker-compose -f docker-compose-gpu.yml up

  # For CPU inference
  docker-compose -f docker-compose-cpu.yaml up
  ```

  This will start the Orpheus TTS service on port 5005. Make sure to update your `.env` file accordingly:

  ```
  BASE_URL=http://localhost:5005/v1
  ```

- Create a .env file from .env_sample and configure it with the correct values. Make sure you follow the instructions mentioned at the top of .env_sample to avoid errors.
  ```bash
  cp .env_sample .env
  ```
- After this, choose between the below options for the next step to run the audiobook creator app:

   <details>
   <summary>Quickest Start (docker run)</summary>

  - Make sure your .env is configured correctly and your LLM and Kokoro FastAPI are running. In the same folder where .env is present, run the below command
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
       ghcr.io/prakharsr/audiobook_creator_gpu:v1.3
    ```

    For CPU based inference

    ```bash
    docker run \
       --name audiobook_creator \
       --restart always \
       --network host \
       --env-file .env \
       -v model_cache:/app/model_cache \
       ghcr.io/prakharsr/audiobook_creator_cpu:v1.3
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
  - **Optional TTS Services**: You can optionally include TTS services in your docker-compose setup:
    - To include **Kokoro TTS**: Use `docker compose --profile kokoro up --build`
    - To run **without any TTS service** (if you're running TTS separately): Use `docker compose up --build`
  - Copy the .env file into the audiobook-creator folder
  - Choose between the types of inference:

    For CUDA based GPU inference (Apple Silicon GPUs currently not supported, use CPU based inference instead). Choose the value of MAX_PARALLEL_REQUESTS_BATCH_SIZE based on [this guide](https://github.com/prakharsr/audiobook-creator/?tab=readme-ov-file#parallel-batch-inferencing-of-audio-for-faster-audio-generation) and set the value in fastapi service and env variable.

    ```bash
    cd docker/gpu

    # To include Kokoro TTS service
    docker compose -f docker-compose-kokoro.yml up

    # To run without TTS service (if using external TTS)
    docker compose -f docker-compose-external.yml up
    ```

    For CPU based inference. In this case you can keep number of workers as 1 as only mostly GPU based inferencing benefits from parallel workers and batch requests.

    ```bash
    cd docker/cpu

    # To include Kokoro TTS service
    docker compose --profile kokoro up --build

    # To run without TTS service (if using external TTS)
    docker compose up --build
    ```

  - Wait for the models to download and then navigate to http://localhost:7860 for the Gradio UI
  </details>

   <details>
   <summary>Direct run (via uv)</summary>

  1.  Clone the repository

      ```bash
      git clone https://github.com/prakharsr/audiobook-creator.git

      cd audiobook-creator
      ```

  2.  Make sure your .env is configured correctly and your LLM and TTS service (Kokoro or Orpheus) are running
  3.  Copy the .env file into the audiobook-creator folder
  4.  Install uv
      ```bash
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```
  5.  Create a virtual environment with Python 3.12:
      ```bash
      uv venv --python 3.12
      ```
  6.  Activate the virtual environment:
      ```bash
      source .venv/bin/activate
      ```
  7.  Install Pip 24.0:
      ```bash
      uv pip install pip==24.0
      ```
  8.  Install dependencies (choose CPU or GPU version):
      ```bash
      uv pip install -r requirements_cpu.txt
      ```
      ```bash
      uv pip install -r requirements_gpu.txt
      ```
  9.  Upgrade version of six to avoid errors:
      ```bash
      uv pip install --upgrade six==1.17.0
      ```
  10. Install [calibre](https://calibre-ebook.com/download) (Optional dependency, needed if you need better text decoding capabilities, wider compatibility and want to create M4B audiobook). Also make sure that calibre is present in your PATH. For MacOS, do the following to add it to the PATH:
      ```bash
      deactivate
      echo 'export PATH="/Applications/calibre.app/Contents/MacOS:$PATH"' >> .venv/bin/activate
      source .venv/bin/activate
      ```
  11. Install [ffmpeg](https://www.ffmpeg.org/download.html) (Needed for audio output format conversion and if you want to create M4B audiobook)
  12. In the activated virtual environment, run `uvicorn app:app --host 0.0.0.0 --port 7860` to run the Gradio app. After the app has started, navigate to `http://127.0.0.1:7860` in the browser.
  </details>

### TTS Service Options

This audiobook creator supports multiple TTS (Text-to-Speech) services:

**Kokoro TTS** ([Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI))

- High-quality, fast TTS with multiple voices
- Excellent for quick audiobook generation
- Default port: 8880

**Orpheus TTS** ([Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI))

- High-performance TTS with emotion tags
- 8 different voices with emotional expressions
- Optimized for RTX GPUs
- Default port: 5005

You can choose either service based on your preferences. Both are compatible with the audiobook creator and offer OpenAI-compatible APIs.

### Parallel batch inferencing of audio for faster audio generation

- Choose the value of **MAX_PARALLEL_REQUESTS_BATCH_SIZE** based on your available VRAM to accelerate the generation of audio by using parallel batch inferencing. This variable is used while setting up the number of workers in the TTS docker container and as an env variable for defining the max number of parallel requests that can be made to the TTS service, so make sure you set the same values for both of them. You can consider setting this value to your available (VRAM/ 2) and play around with the value to see if it works best. If you are unsure then a good starting point for this value can be a value of 2. If you face issues of running out of memory then consider lowering the value for both workers and for the env variable.

## Roadmap

Planned future enhancements:

- ⏳ Add support for choosing between various languages which are currently supported by Kokoro.
- ⏳ Add support for [Zonos](https://github.com/Zyphra/Zonos), Models: https://huggingface.co/Zyphra/Zonos-v0.1-hybrid, https://huggingface.co/Zyphra/Zonos-v0.1-transformer. Zonos supports voices with a wide range of emotions so adding that as a feature will greatly enhance the listening experience.
- ✅ Add support for [Orpheus TTS](https://github.com/Lex-au/Orpheus-FastAPI). Orpheus supports emotion tags and 8 different voices for a more immersive listening experience.
- ✅ Support batch inference for Kokoro to speed up audiobook generation
- ✅ Give choice to the user to select the voice in which they want the book to be read (male voice/ female voice)
- ✅ Add support for running the app through docker.
- ✅ Create UI using Gradio.
- ✅ Try different voice combinations using `generate_audio_samples.py` and update the `voice_map.json` to use better voices.
- ✅ Add support for the these output formats: AAC, M4A, MP3, WAV, OPUS, FLAC, PCM, M4B.
- ✅ Add support for using calibre to extract the text and metadata for better formatting and wider compatibility.
- ✅ Add artwork and chapters, and convert audiobooks to M4B format for better compatibility.
- ✅ Give option to the user for selecting the audio generation format.
- ✅ Add extended pause when chapters end once chapter recognition is in place.
- ✅ Improve single-voice narration with a different dialogue voice from the narrator's voice.
- ✅ Read out only the dialogue in a different voice instead of the entire line in that voice.

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
