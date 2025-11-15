# Audiobook Creator

## Overview

Audiobook Creator is an open-source project designed to convert books in various text formats (e.g., EPUB, PDF, etc.) into fully voiced audiobooks with intelligent character voice attribution. It leverages modern Large Language Models (LLMs), and Text-to-Speech (TTS) technologies to create an engaging and dynamic audiobook experience. The project features professional-grade TTS engines including [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) and [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) with advanced async parallel processing, offering exceptional audio quality with memory-efficient processing. The project is licensed under the GNU General Public License v3.0 (GPL-3.0), ensuring that it remains free and open for everyone to use, modify, and distribute.

Sample multi voice audio for a short story generated using *Orpheus TTS* (with added emotions and better sounding audio) : https://audio.com/prakhar-sharma/audio/sample-orpheus-multi-voice-audiobook-orpheus

Sample multi voice audio for a short story generated using *Kokoro TTS* : https://audio.com/prakhar-sharma/audio/generated-sample-multi-voice-audiobook-kokoro

Watch the demo video:

[![Watch the demo video](https://img.youtube.com/vi/E5lUQoBjquo/maxresdefault.jpg)](https://www.youtube.com/watch?v=E5lUQoBjquo)

<details>
<summary>The project consists of four main components:</summary>

1. **Text Cleaning and Formatting (`book_to_txt.py`)**:
   - Extracts and cleans text from a book file (e.g., `book.epub`).
   - Normalizes special characters, fixes line breaks, and corrects formatting issues such as unterminated quotes or incomplete lines.
   - Outputs the cleaned text to `converted_book.txt`.

2. **Character Identification and Metadata Generation (`identify_characters_and_output_book_to_jsonl.py`)**:
   - Step 1: Identifies unique characters and their age, gender in the text using an LLM via an OpenAI-compatible API.
   - Step 2: Attributes the speakers identified in each line of text using the characters identified in step 1 using an LLM via an OpenAI-compatible API.
   - Outputs two files:
     - `speaker_attributed_book.jsonl`: Each line of text annotated with the identified speaker.
     - `character_gender_map.json`: Metadata about characters, including name, age, gender, and gender score.

3. **Emotion Tags Enhancement (`add_emotion_tags.py`)**:
   - Adds emotion tags (e.g., `<laugh>`, `<sigh>`, `<gasp>` etc.) to enhance narration expressiveness.
   - Processes `converted_book.txt` and outputs enhanced text to `tag_added_lines_chunks.txt`.
   - Requires Orpheus TTS engine for emotion tag support.

4. **Audiobook Generation (`generate_audiobook.py`)**:
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
- **Character Identification**: Identifies characters and infers their attributes (gender, age) then attributes speakers to each line of text using LLMs.
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
- Set up your LLM and expose an OpenAI-compatible endpoint (e.g., using LM Studio with `Qwen/Qwen3-30B-A3B-Instruct-2507`).
- **Set up TTS Engine**: Choose between Kokoro or Orpheus TTS models:

   ### Option 1: Kokoro TTS (Recommended for most users)

   <details>
   <summary>
   Expand to view
   </summary>

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

   </details>

   ### Option 2: Orpheus TTS (High-Quality and More Expressive Audio and Support for Emotion Tags)

   <details>
   <summary>
   Expand to view
   </summary>

   Experience premium audio quality with the Orpheus TTS FastAPI server featuring vLLM backend and bfloat16/ float16/ float32 precision. Set up Orpheus TTS using my dedicated [Orpheus TTS FastAPI](https://github.com/prakharsr/Orpheus-TTS-FastAPI) repository.

   **IMPORTANT:** Choose only highest possible precision (bf16/ fp16/ fp32) and this vLLM based FastAPI server as I noticed that using quantized versions of Orpheus or even using float16 GGUF with llama.cpp gave me audio quality issues and artifacts (repeated lines in audio/ extended audio with no spoken text but weird noises/ audio hallucinations/ infinite audio looping/ some other issues). The linked repository for FastAPI server also has some additional improvements to fix such issues by detecting decoding errors, infinite audio loops and having a retry mechanism which tries to fix these audio issues automatically. 

   **Setup Instructions:**
   Please follow the complete setup instructions from the [Orpheus TTS FastAPI repository](https://github.com/prakharsr/Orpheus-TTS-FastAPI) as it contains all the necessary configuration details, installation steps, and optimization guides.

   </details>
<br/>

- Create a .env file from .env_sample and configure it with the correct values. Make sure you follow the instructions mentioned at the top of .env_sample to avoid errors.
   ```bash
   cp .env_sample .env
   ```
- After this, choose between the below options for the next step to run the audiobook creator app: 

   **Quickest Start (docker run)**

   - Make sure your .env is configured correctly and your LLM and TTS engine (Kokoro/Orpheus) are running. In the same folder where .env is present, run the below command

      ```bash
      docker run \
         --name audiobook_creator \
         --restart always \
         --network host \
         --gpus all \
         --env-file .env \
         ghcr.io/prakharsr/audiobook_creator:v2.0
      ```

   - Navigate to http://localhost:7860 for the Gradio UI

   <details>
   <summary>Quick Start (docker compose)</summary>

   - Clone the repository
      ```bash 
      git clone https://github.com/prakharsr/audiobook-creator.git

      cd audiobook-creator
      ```
   - Make sure your .env is configured correctly and your LLM is running
   - For Kokoro TTS, please refer to the instructions mentioned before in Option 1
   - For Orpheus TTS, please refer to the [Orpheus TTS FastAPI repository](https://github.com/prakharsr/Orpheus-TTS-FastAPI) for setup instructions as it provides the optimal implementation.
   - Copy the .env file into the audiobook-creator folder
   -  Run using
      ```
      docker compose --env-file .env up --build
      ```
   - Navigate to http://localhost:7860 for the Gradio UI
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
   6. Install dependedncies:
      ```bash
      uv pip install -r requirements.txt --no-deps
      ```
   7. Install [calibre](https://calibre-ebook.com/download) (Optional dependency, needed if you need better text decoding capabilities, wider compatibility and want to create M4B audiobook). Also make sure that calibre is present in your PATH. For MacOS, do the following to add it to the PATH:
      ```bash
      deactivate
      echo 'export PATH="/Applications/calibre.app/Contents/MacOS:$PATH"' >> .venv/bin/activate
      source .venv/bin/activate
      ```
   8. Install [ffmpeg](https://www.ffmpeg.org/download.html) (Needed for audio output format conversion and if you want to create M4B audiobook)
   9. In the activated virtual environment, run `uvicorn app:app --host 0.0.0.0 --port 7860` to run the Gradio app. After the app has started, navigate to `http://127.0.0.1:7860` in the browser.
   </details>


### Context window size and parallel requests config for character identification and emotion tag addition

<details>
<summary>
Expand to view
</summary>

#### For Character Identification Step

- This step creates batches of data to identify unique characters and then attributes speakers to each line of text. So, this requires lots of context so I recommend having atleast 20,000 context window size for non-thinking models, you may require more context size if you're using a thinking model. Also, this step is sequential for maximum accuracy so you can use set parallel/ max-num-seqs/ max-running-requests to 1 in llama.cpp/ vllm/ sglang to reduce your memory footprint.

#### For Emotion Tag Addition Step

- This step intelligently identifies where emotion tag might be needed to be added and then creates small batches to send to LLM which then returns the emotion tag added text. This step requires atleast 8192 context window with a thinking llm for each request. Also, this step can be parallelized since each batch is independent of each other, so you can make use of parallel/ max-num-seqs/ max-running-requests parameters in llama.cpp/ vllm/ sglang. Make sure that while using parallel param in llama.cpp, you also increase the context window size with the same factor since in llama.cpp, the context is split for each parallel request. So, if you're setting parallel to 8 in llama.cpp then set context size to 8 x 8192 = 65536. 

</details>

### Parallel batch inferencing of audio for faster audio generation

<details>
<summary>
Expand to view
</summary>

Choose the value of **TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE** based on your available VRAM to accelerate the generation of audio by using parallel batch inferencing. This variable defines the max number of parallel requests that can be made to TTS FastAPI for faster audio generation.

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

</details>

## Roadmap

Planned future enhancements:

-  ⏳ Add support for choosing between various languages which are currently supported by Kokoro and Orpheus.
-  ✅ Remove Gliner NLP pipeline for character identification and build two step LLM based character identification pipeline for maximum accuracy.
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
