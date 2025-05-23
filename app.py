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
import glob
import json
import time
from datetime import datetime
from fastapi import FastAPI
from book_to_txt import process_book_and_extract_text, save_book
from identify_characters_and_output_book_to_jsonl import (
    process_book_and_identify_characters,
)
from generate_audiobook import process_audiobook_generation

css = """
.step-heading {font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem}
"""

app = FastAPI()

# Simple task tracking for background processes
TASKS_FILE = "generation_tasks.json"


def load_tasks():
    """Load current tasks from file"""
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {}


def save_tasks(tasks):
    """Save tasks to file"""
    try:
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f)
    except:
        pass


def update_task_status(task_id, status, progress="", error=None):
    """Update task status"""
    tasks = load_tasks()
    tasks[task_id] = {
        "status": status,
        "progress": progress,
        "timestamp": datetime.now().isoformat(),
        "error": error,
    }
    save_tasks(tasks)


def get_active_tasks():
    """Get list of active generation tasks"""
    tasks = load_tasks()
    active_tasks = []
    for task_id, task_info in tasks.items():
        if task_info.get("status") in ["running", "starting"]:
            # Check if task is still actually running (simple heuristic)
            timestamp = datetime.fromisoformat(task_info["timestamp"])
            if (datetime.now() - timestamp).seconds < 300:  # 5 minutes timeout
                active_tasks.append(
                    {
                        "id": task_id,
                        "progress": task_info.get("progress", ""),
                        "timestamp": task_info["timestamp"],
                    }
                )
    return active_tasks


def get_past_generated_files():
    """Get list of past generated audiobook files"""
    try:
        audiobooks_dir = "generated_audiobooks"
        if not os.path.exists(audiobooks_dir):
            return []

        # Get all files in the generated_audiobooks directory
        pattern = os.path.join(audiobooks_dir, "*")
        files = glob.glob(pattern)

        # Filter out directories and get file info
        audiobook_files = []
        for file_path in files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                # Get file size and modification time
                stat_info = os.stat(file_path)
                size_mb = round(stat_info.st_size / (1024 * 1024), 2)
                mod_time = datetime.fromtimestamp(stat_info.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )

                audiobook_files.append(
                    {
                        "path": file_path,
                        "filename": filename,
                        "size_mb": size_mb,
                        "modified": mod_time,
                    }
                )

        # Sort by modification time (newest first)
        audiobook_files.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
        return audiobook_files
    except Exception as e:
        print(f"Error getting past files: {e}")
        return []


def delete_audiobook_file(file_path):
    """Delete an audiobook file and return status message"""
    if not file_path:
        return gr.Warning("No file selected for deletion.")

    try:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            file_size = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            os.remove(file_path)
            return gr.Info(
                f"Successfully deleted '{filename}' ({file_size} MB).", duration=5
            )
        else:
            return gr.Warning("File not found. It may have already been deleted.")
    except PermissionError:
        return gr.Warning("Permission denied. Cannot delete the file.")
    except Exception as e:
        return gr.Warning(f"Error deleting file: {str(e)}")


def get_selected_file_info(file_path):
    """Get information about the selected file for confirmation"""
    if not file_path:
        return "No file selected."

    try:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            file_size = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime(
                "%Y-%m-%d %H:%M"
            )
            return f"**Selected for deletion:**\n\n📁 **{filename}**\n📊 Size: {file_size} MB\n📅 Modified: {mod_time}\n\n⚠️ **This action cannot be undone!**"
        else:
            return "File not found."
    except Exception as e:
        return f"Error reading file info: {str(e)}"


def refresh_past_files():
    """Refresh the list of past generated files"""
    files = get_past_generated_files()
    active_tasks = get_active_tasks()

    # Build display text
    display_parts = []

    # Show active tasks first
    if active_tasks:
        display_parts.append("### 🔄 Currently Running:")
        for task in active_tasks:
            task_display = f"⏳ **Generation in progress** - {task['progress']} ({task['timestamp'][:16]})"
            display_parts.append(task_display)
        display_parts.append("")  # Empty line

    # Show past files
    if not files:
        if not active_tasks:
            display_parts.append("No past audiobooks found.")
    else:
        if active_tasks:
            display_parts.append("### 📁 Completed Audiobooks:")

        file_choices = []
        for file_info in files:
            file_display = f"📁 **{file_info['filename']}** ({file_info['size_mb']} MB) - {file_info['modified']}"
            display_parts.append(file_display)
            file_choices.append((file_info["filename"], file_info["path"]))

        file_text = "\n\n".join(display_parts)
        return (
            gr.update(value=file_text, visible=True),
            gr.update(
                choices=file_choices,
                value=file_choices[0][1] if file_choices else None,
                visible=bool(file_choices),
            ),
            gr.update(visible=bool(file_choices)),
        )  # Show delete button if files exist

    file_text = "\n\n".join(display_parts)
    return (
        gr.update(value=file_text, visible=True),
        gr.update(choices=[], value=None, visible=False),
        gr.update(visible=False),
    )  # Hide delete button if no files


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
        return gr.Info(
            "Text extracted successfully! You can now edit the content.", duration=5
        )
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
        return gr.Info(
            "📖 Book saved successfully as 'converted_book.txt'!", duration=10
        )
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
        yield gr.Info(
            "Character identification complete! Proceed to audiobook generation.",
            duration=5,
        )
        yield last_output
        return
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield gr.Warning(f"Error identifying characters: {str(e)}")
        yield None
        return


async def generate_audiobook_wrapper(
    voice_type, narrator_gender, output_format, book_file, book_title
):
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

    # Create a unique task ID for tracking
    task_id = f"audiobook_{int(time.time())}_{book_title[:20].replace(' ', '_')}"

    try:
        # Mark task as starting
        update_task_status(
            task_id,
            "starting",
            f"Initializing {voice_type} audiobook generation for '{book_title}'",
        )

        last_output = None
        audiobook_path = None
        # Pass through all yield values from the original function
        async for output in process_audiobook_generation(
            voice_type, narrator_gender, output_format, book_file, book_title
        ):
            last_output = output
            # Update task status with current progress
            update_task_status(task_id, "running", output)
            yield output, None  # Yield each progress update without file path

        # Get the correct file extension based on the output format
        generate_m4b_audiobook_file = (
            True if output_format == "M4B (Chapters & Cover)" else False
        )
        file_extension = "m4b" if generate_m4b_audiobook_file else output_format.lower()

        # Set the audiobook file path according to the provided information
        # Sanitize book title for filename
        safe_book_title = "".join(
            c for c in book_title if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_book_title = (
            safe_book_title or "audiobook"
        )  # fallback if title becomes empty
        audiobook_path = os.path.join(
            "generated_audiobooks", f"{safe_book_title}.{file_extension}"
        )

        # Mark task as completed
        update_task_status(
            task_id,
            "completed",
            f"Audiobook generated successfully: {safe_book_title}.{file_extension}",
        )

        # Final yield with success notification and file path
        yield gr.Info(
            f"Audiobook generated successfully in {output_format} format! You can now download it in the Download section. Click on the blue download link next to the file name. If you lost connection during generation, check the 'Past Generated Audiobooks' section.",
            duration=15,
        ), None
        yield last_output, audiobook_path
        return
    except Exception as e:
        # Mark task as failed
        update_task_status(task_id, "failed", "Generation failed", str(e))
        print(e)
        traceback.print_exc()
        yield gr.Warning(f"Error generating audiobook: {str(e)}"), None
        yield None, None
        return


with gr.Blocks(css=css, theme=gr.themes.Default()) as gradio_app:
    gr.Markdown("# 📖 Audiobook Creator")
    gr.Markdown("Create professional audiobooks from your ebooks in just a few steps.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('<div class="step-heading">📚 Step 1: Book Details</div>')

            book_title = gr.Textbox(
                label="Book Title",
                placeholder="Enter the title of your book",
                info="This will be used for finding the protagonist of the book in the character identification step",
            )

            book_input = gr.File(label="Upload Book")

            text_decoding_option = gr.Radio(
                ["textract", "calibre"],
                label="Text Extraction Method",
                value="textract",
                info="Use calibre for better formatted results, wider compatibility for ebook formats. You can try both methods and choose based on the output result.",
            )

            validate_btn = gr.Button("Validate Book", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                '<div class="step-heading">✂️ Step 2: Extract & Edit Content</div>'
            )

            convert_btn = gr.Button("Extract Text", variant="primary")

            with gr.Accordion("Editing Tips", open=True):
                gr.Markdown(
                    """
                * Remove unwanted sections: Table of Contents, About the Author, Acknowledgements
                * Fix formatting issues or OCR errors
                * Check for chapter breaks and paragraph formatting
                """
                )

            text_output = gr.Textbox(
                label="Edit Book Content",
                placeholder="Extracted text will appear here for editing",
                interactive=True,
                lines=15,
            )

            save_btn = gr.Button("Save Edited Text", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                '<div class="step-heading">🧩 Step 3: Character Identification (Optional)</div>'
            )

            identify_btn = gr.Button("Identify Characters", variant="primary")

            with gr.Accordion("Why Identify Characters?", open=True):
                gr.Markdown(
                    """
                * Improves multi-voice narration by assigning different voices to characters
                * Creates more engaging audiobooks with distinct character voices
                * Skip this step if you prefer single-voice narration
                """
                )

            character_output = gr.Textbox(
                label="Character Identification Progress",
                placeholder="Character identification progress will be shown here",
                interactive=False,
                lines=3,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">🎧 Step 4: Generate Audiobook</div>')

            with gr.Row():
                voice_type = gr.Radio(
                    ["Single Voice", "Multi-Voice"],
                    label="Narration Type",
                    value="Single Voice",
                    info="Multi-Voice requires character identification",
                )

                narrator_gender = gr.Radio(
                    ["male", "female"],
                    label="Choose whether you want the book to be read in a male or female voice",
                    value="female",
                )

                output_format = gr.Dropdown(
                    [
                        "M4B (Chapters & Cover)",
                        "AAC",
                        "M4A",
                        "MP3",
                        "WAV",
                        "OPUS",
                        "FLAC",
                        "PCM",
                    ],
                    label="Output Format",
                    value="M4B (Chapters & Cover)",
                    info="M4B supports chapters and cover art",
                )

            generate_btn = gr.Button("Generate Audiobook", variant="primary")

            audio_output = gr.Textbox(
                label="Generation Progress",
                placeholder="Generation progress will be shown here",
                interactive=False,
                lines=3,
            )

            # Add a new File component for downloading the audiobook
            with gr.Group(visible=False) as download_box:
                gr.Markdown("### 📥 Download Your Audiobook")
                audiobook_file = gr.File(
                    label="Download Generated Audiobook",
                    interactive=False,
                    type="filepath",
                )

        # Past generated files section
    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">📂 Generated Audiobooks</div>')

            with gr.Row():
                refresh_btn = gr.Button(
                    "🔄 Refresh List", variant="secondary", size="sm"
                )

            past_files_display = gr.Markdown(
                value="Click refresh to see generated audiobooks.", visible=True
            )

            past_files_dropdown = gr.Dropdown(
                label="Select a past audiobook",
                choices=[],
                visible=False,
                interactive=True,
            )

            with gr.Row():
                past_file_download = gr.File(
                    label="Download Selected Audiobook",
                    interactive=False,
                    type="filepath",
                    visible=False,
                )

                delete_btn = gr.Button(
                    "🗑️ Delete Selected",
                    variant="stop",
                    size="sm",
                    visible=False,
                )

            # Confirmation dialog for deletion (hidden by default)
            with gr.Group(visible=False) as delete_confirmation:
                gr.Markdown("### ⚠️ Confirm Deletion")
                delete_info_display = gr.Markdown("")

                with gr.Row():
                    confirm_delete_btn = gr.Button(
                        "🗑️ Yes, Delete", variant="stop", size="sm"
                    )
                    cancel_delete_btn = gr.Button(
                        "❌ Cancel", variant="secondary", size="sm"
                    )

    # Connections with proper handling of Gradio notifications
    validate_btn.click(
        validate_book_upload, inputs=[book_input, book_title], outputs=[]
    )

    convert_btn.click(
        text_extraction_wrapper,
        inputs=[book_input, text_decoding_option, book_title],
        outputs=[text_output],
        queue=True,
    )

    save_btn.click(
        save_book_wrapper, inputs=[text_output, book_title], outputs=[], queue=True
    )

    identify_btn.click(
        identify_characters_wrapper,
        inputs=[book_title],
        outputs=[character_output],
        queue=True,
    )

    # Update the generate_audiobook_wrapper to output both progress text and file path
    generate_btn.click(
        generate_audiobook_wrapper,
        inputs=[voice_type, narrator_gender, output_format, book_input, book_title],
        outputs=[audio_output, audiobook_file],
        queue=True,
    ).then(
        # Make the download box visible after generation completes successfully
        lambda x: (
            gr.update(visible=True) if x is not None else gr.update(visible=False)
        ),
        inputs=[audiobook_file],
        outputs=[download_box],
    ).then(
        # Refresh past files list after successful generation
        lambda x: (
            refresh_past_files()
            if x is not None
            else (gr.update(), gr.update(), gr.update())
        ),
        inputs=[audiobook_file],
        outputs=[past_files_display, past_files_dropdown, delete_btn],
    )

    # Refresh past files when the refresh button is clicked
    refresh_btn.click(
        refresh_past_files,
        outputs=[past_files_display, past_files_dropdown, delete_btn],
    )

    # Update the download file when a past file is selected
    past_files_dropdown.change(
        lambda x: gr.update(value=x, visible=True) if x else gr.update(visible=False),
        inputs=[past_files_dropdown],
        outputs=[past_file_download],
    )

    # Show confirmation dialog when delete button is clicked
    delete_btn.click(
        get_selected_file_info,
        inputs=[past_files_dropdown],
        outputs=[delete_info_display],
    ).then(lambda: gr.update(visible=True), outputs=[delete_confirmation])

    # Actually delete the file when confirmed
    confirm_delete_btn.click(
        delete_audiobook_file, inputs=[past_files_dropdown], outputs=[]
    ).then(
        # Hide confirmation dialog
        lambda: gr.update(visible=False),
        outputs=[delete_confirmation],
    ).then(
        # Refresh the list after deletion
        refresh_past_files,
        outputs=[past_files_display, past_files_dropdown, delete_btn],
    ).then(
        # Clear the download file after deletion
        lambda: gr.update(value=None, visible=False),
        outputs=[past_file_download],
    )

    # Cancel deletion
    cancel_delete_btn.click(
        lambda: gr.update(visible=False), outputs=[delete_confirmation]
    )

    # Load past files when the app starts
    gradio_app.load(
        refresh_past_files,
        outputs=[past_files_display, past_files_dropdown, delete_btn],
    )

app = gr.mount_gradio_app(app, gradio_app, path="/")  # Mount Gradio at root

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
