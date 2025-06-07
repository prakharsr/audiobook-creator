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

import gradio as gr
import os
import traceback
from fastapi import FastAPI
from book_to_txt import process_book_and_extract_text, save_book
from identify_characters_and_output_book_to_jsonl import process_book_and_identify_characters
from generate_audiobook import process_audiobook_generation
from dotenv import load_dotenv

load_dotenv()

css = """
.step-heading {font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem}
"""

app = FastAPI()

def validate_book_upload(book_file, book_title):
    """Validate book upload and return a notification"""
    if book_file is None:
        return gr.Warning("Please upload a book file first.")
    
    if not book_title:
        return gr.Warning("Please enter a book title.")
    
    return gr.Info(f"Book '{book_title}' ready for processing.", duration=5)

def text_extraction_wrapper(book_file, text_decoding_option, book_title):
    """Wrapper for text extraction with validation and progress updates"""
    if book_file is None or not book_title:
        yield None
        return gr.Warning("Please upload a book file and enter a title first.")
    
    try:
        last_output = None
        # Pass through all yield values from the original function
        for output in process_book_and_extract_text(book_file, text_decoding_option):
            last_output = output
            yield output  # Yield each progress update
        
        # Final yield with success notification
        yield last_output
        return gr.Info("Text extracted successfully! You can now edit the content.", duration=5)
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield None
        return gr.Warning(f"Error extracting text: {str(e)}")

def save_book_wrapper(text_content, book_title):
    """Wrapper for saving book with validation"""
    if not text_content:
        return gr.Warning("No text content to save.")
    
    if not book_title:
        return gr.Warning("Please enter a book title before saving.")
    
    try:
        save_book(text_content)
        return gr.Info("📖 Book saved successfully as 'converted_book.txt'!", duration=10)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.Warning(f"Error saving book: {str(e)}")

async def identify_characters_wrapper(book_title):
    """Wrapper for character identification with validation and progress updates"""
    if not book_title:
        yield gr.Warning("Please enter a book title first.")
        yield None
        return

    try:
        last_output = None
        # Pass through all yield values from the original function
        async for output in process_book_and_identify_characters(book_title):
            last_output = output
            yield output  # Yield each progress update
        
        # Final yield with success notification
        yield gr.Info("Character identification complete! Proceed to audiobook generation.", duration=5)
        yield last_output
        return
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield gr.Warning(f"Error identifying characters: {str(e)}")
        yield None
        return

async def generate_audiobook_wrapper(voice_type, narrator_gender, output_format, book_file, book_title, add_emotion_tags):
    """Wrapper for audiobook generation with validation and progress updates"""
    if book_file is None:
        yield gr.Warning("Please upload a book file first."), None
        yield None, None
        return
    if not book_title:
        yield gr.Warning("Please enter a book title first."), None
        yield None, None
        return
    if not voice_type or not output_format:
        yield gr.Warning("Please select voice type and output format."), None
        yield None, None
        return
    
    # Get current TTS engine from environment
    current_tts_engine = os.environ.get("TTS_MODEL", "kokoro").lower()
    
    # Validate emotion tags selection
    if add_emotion_tags and current_tts_engine != "orpheus":
        yield gr.Warning("Emotion tags are only supported with Orpheus TTS engine."), None
        yield None, None
        return
        
    try:
        last_output = None
        audiobook_path = None
        # Pass through all yield values from the original function
        async for output in process_audiobook_generation(voice_type, narrator_gender, output_format, book_file, add_emotion_tags):
            last_output = output
            yield output, None  # Yield each progress update without file path
        
        # Get the correct file extension based on the output format
        generate_m4b_audiobook_file = True if output_format == "M4B (Chapters & Cover)" else False
        file_extension = "m4b" if generate_m4b_audiobook_file else output_format.lower()
        
        # Set the audiobook file path according to the provided information
        audiobook_path = os.path.join("generated_audiobooks", f"audiobook.{file_extension}")
        
        # Final yield with success notification and file path
        yield gr.Info(f"Audiobook generated successfully in {output_format} format! You can now download it in the Download section. Click on the blue download link next to the file name.", duration=10), None
        yield last_output, audiobook_path
        return
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield gr.Warning(f"Error generating audiobook: {str(e)}"), None
        yield None, None
        return

with gr.Blocks(css=css, theme=gr.themes.Default()) as gradio_app:
    gr.Markdown("# 📖 Audiobook Creator")
    gr.Markdown("Create professional audiobooks from your ebooks in just a few steps.")
    
    # Get TTS configuration from environment variables
    current_tts_engine = os.environ.get("TTS_MODEL", "kokoro").lower()
    tts_base_url = os.environ.get("TTS_BASE_URL", "Not configured")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('<div class="step-heading">📚 Step 1: Book Details</div>')
            
            book_title = gr.Textbox(
                label="Book Title", 
                placeholder="Enter the title of your book",
                info="This will be used for finding the protagonist of the book in the character identification step"
            )
            
            book_input = gr.File(
                label="Upload Book"
            )
            
            text_decoding_option = gr.Radio(
                ["textract", "calibre"], 
                label="Text Extraction Method", 
                value="textract",
                info="Use calibre for better formatted results, wider compatibility for ebook formats. You can try both methods and choose based on the output result."
            )
            
            validate_btn = gr.Button("Validate Book", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">✂️ Step 2: Extract & Edit Content</div>')
            
            convert_btn = gr.Button("Extract Text", variant="primary")
            
            with gr.Accordion("Editing Tips", open=True):
                gr.Markdown("""
                * Remove unwanted sections: Table of Contents, About the Author, Acknowledgements
                * Fix formatting issues or OCR errors
                * Check for chapter breaks and paragraph formatting
                """)
            
            text_output = gr.Textbox(
                label="Edit Book Content", 
                placeholder="Extracted text will appear here for editing",
                interactive=True, 
                lines=15
            )
            
            save_btn = gr.Button("Save Edited Text", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">🧩 Step 3: Character Identification (Optional - Requires LLM)</div>')
            
            identify_btn = gr.Button("Identify Characters", variant="primary")
            
            with gr.Accordion("Why Identify Characters?", open=True):
                gr.Markdown("""
                * Improves multi-voice narration by assigning different voices to characters
                * Creates more engaging audiobooks with distinct character voices
                * Skip this step if you prefer single-voice narration
                """)
                
            character_output = gr.Textbox(
                label="Character Identification Progress", 
                placeholder="Character identification progress will be shown here",
                interactive=False,
                lines=3
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">🎧 Step 4: Generate Audiobook</div>')
            
            with gr.Row():
                voice_type = gr.Radio(
                    ["Single Voice", "Multi-Voice"], 
                    label="Narration Type",
                    value="Single Voice",
                    info="Multi-Voice requires character identification"
                )

                narrator_gender = gr.Radio(
                    ["male", "female"], 
                    label="Choose whether you want the book to be read in a male or female voice",
                    value="female"
                )
                
                tts_engine_display = gr.Radio(
                    ["kokoro", "orpheus"], 
                    label="TTS Engine",
                    value=current_tts_engine,
                    interactive=False,
                    info="Configure TTS engine in .env file. Orpheus supports emotion tags."
                )
                
                output_format = gr.Dropdown(
                    ["M4B (Chapters & Cover)", "AAC", "M4A", "MP3", "WAV", "OPUS", "FLAC", "PCM"], 
                    label="Output Format",
                    value="M4B (Chapters & Cover)",
                    info="M4B supports chapters and cover art"
                )
            
            # Emotion tags option (conditional visibility based on TTS engine in .env)
            emotion_tags_visible = current_tts_engine == "orpheus"
            with gr.Group(visible=emotion_tags_visible) as emotion_tags_group:
                emotion_tags_enabled = gr.Checkbox(
                    label="Automatically add emotion tags to the text to enhance narration ? (Optional - Requires LLM)",
                    value=False
                )
                
                with gr.Accordion("About Emotion Tags", open=True):
                    gr.Markdown("""
                    **Emotion Tags enhance your audiobook by adding natural expressions:**

                    * **`<laugh>`** - For laughter or when text indicates laughing
                    * **`<chuckle>`** - For light laughter or chuckling sounds
                    * **`<sigh>`** - For sighing or expressions of resignation/relief
                    * **`<cough>`** - For coughing sounds or throat clearing
                    * **`<sniffle>`** - For sniffling or nasal sounds (emotion, cold, etc.)
                    * **`<groan>`** - For groaning sounds expressing discomfort/frustration
                    * **`<yawn>`** - For yawning or expressions of tiredness
                    * **`<gasp>`** - For gasping sounds of surprise/shock
                    
                    These tags are automatically placed based on the text context and work only with **Orpheus TTS**.
                    """)
            
            generate_btn = gr.Button("Generate Audiobook", variant="primary")
            
            audio_output = gr.Textbox(
                label="Generation Progress", 
                placeholder="Generation progress will be shown here",
                interactive=False,
                lines=3
            )
            
            # Add a new File component for downloading the audiobook
            with gr.Group(visible=False) as download_box:
                gr.Markdown("### 📥 Download Your Audiobook")
                audiobook_file = gr.File(
                    label="Download Generated Audiobook",
                    interactive=False,
                    type="filepath"
                )
    
    # Connections with proper handling of Gradio notifications
    validate_btn.click(
        validate_book_upload, 
        inputs=[book_input, book_title], 
        outputs=[]
    )
    
    convert_btn.click(
        text_extraction_wrapper, 
        inputs=[book_input, text_decoding_option, book_title], 
        outputs=[text_output],
        queue=True
    )
    
    save_btn.click(
        save_book_wrapper, 
        inputs=[text_output, book_title], 
        outputs=[],
        queue=True
    )
    
    identify_btn.click(
        identify_characters_wrapper, 
        inputs=[book_title], 
        outputs=[character_output],
        queue=True
    )
    
    # Update the generate_audiobook_wrapper to output both progress text and file path
    generate_btn.click(
        generate_audiobook_wrapper, 
        inputs=[voice_type, narrator_gender, output_format, book_input, book_title, emotion_tags_enabled], 
        outputs=[audio_output, audiobook_file],
        queue=True
    ).then(
        # Make the download box visible after generation completes successfully
        lambda x: gr.update(visible=True) if x is not None else gr.update(visible=False),
        inputs=[audiobook_file],
        outputs=[download_box]
    )

app = gr.mount_gradio_app(app, gradio_app, path="/")  # Mount Gradio at root

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)