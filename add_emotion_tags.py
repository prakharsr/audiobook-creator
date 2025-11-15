"""
Audiobook Creator - Emotion Tags Module
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
import asyncio
from utils.add_emotion_tags_to_text import add_tags_to_text_chunks, process_emotion_tags_for_jsonl_data
from utils.file_utils import read_jsonl
from dotenv import load_dotenv

load_dotenv()

TTS_MODEL = os.environ.get("TTS_MODEL", "kokoro")

async def process_emotion_tags(characters_identified):
    """
    Process emotion tags for book text enhancement.
    
    This function reads the converted book text and adds emotion tags to enhance narration.
    The enhanced text is saved to tag_added_lines_chunks.txt. This process is voice-agnostic
    and can be used with both single-voice and multi-voice audiobook generation.
        
    Yields:
        str: Progress updates as emotion tags are being processed
    """
    
    # Check if TTS engine supports emotion tags
    if TTS_MODEL.lower() != "orpheus":
        yield f"❌ Emotion tags are only supported with Orpheus TTS engine. Current engine: {TTS_MODEL}"
        return
    
    # Check if converted book exists
    if not os.path.exists("converted_book.txt"):
        yield "❌ No converted book found. Please extract text first."
        return
    
    print(f"Characters identified value: {characters_identified}")
    
    try:
        jsonl = []
        if characters_identified:
            print("Characters identified. Reading speaker attributed book JSONL file...")
            jsonl = read_jsonl("speaker_attributed_book.jsonl")
        else:
            print("Characters not identified. Reading converted book text...")
            # Read the converted book text
            with open("converted_book.txt", "r", encoding='utf-8') as f:
                text = f.read()

                for line in text.split("\n"):
                    jsonl.append({
                        "line": line,
                        "speaker": "narrator"
                    })

        yield f"🎭 Adding emotion tags to the book text..."
        yield "📖 Loaded book text for emotion enhancement"
        
        # Process the text with emotion tags
        async for progress in process_emotion_tags_for_jsonl_data(jsonl):
            yield progress
        
        yield "✅ Emotion tags processing completed successfully!"
        yield "📁 Enhanced text saved to tag_added_lines_chunks.txt"
        yield "ℹ️ Enhanced text can now be used with both single-voice and multi-voice generation"
        
    except Exception as e:
        yield f"❌ Error processing emotion tags: {str(e)}"
        raise e

async def main():
    """
    Main function for standalone execution of emotion tags processing.
    Allows users to run this script directly from command line.
    """
    
    # Check if TTS engine supports emotion tags
    if TTS_MODEL.lower() != "orpheus":
        print(f"❌ Emotion tags are only supported with Orpheus TTS engine. Current engine: {TTS_MODEL}")
        print("Please update your .env file to set TTS_MODEL=orpheus")
        return
    
    print("🎭 **Audiobook Creator - Emotion Tags Enhancement**")
    print("="*60)
    print("📖 This process enhances your book text with emotion tags")
    print("🎯 The enhanced text can be used with any voice configuration")
    print()
    
    try:
        async for progress in process_emotion_tags():
            print(progress)
        
        print("💡 You can now generate audiobooks with enhanced emotional expression!")
        
    except Exception as e:
        print(f"\n❌ Error processing emotion tags: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 