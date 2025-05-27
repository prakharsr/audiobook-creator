import os
from dotenv import load_dotenv
from utils.file_utils import read_json

load_dotenv()

TEMP_DIR = "temp"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8880/v1")
API_KEY = os.environ.get("API_KEY", "not-needed")
MODEL = os.environ.get("MODEL", "kokoro")
MAX_PARALLEL_REQUESTS_BATCH_SIZE = int(
    os.environ.get("MAX_PARALLEL_REQUESTS_BATCH_SIZE", 2)
)
VOICE_MAP = (
    read_json("static_files/kokoro_voice_map_male_narrator.json")
    if MODEL == "kokoro"
    else read_json("static_files/orpheus_voice_map_male_narrator.json")
)
CHAPTER_LIST_FILE = "chapter_list.txt"
FFMPEG_METADATA_FILE = "ffmpeg_metadata.txt"