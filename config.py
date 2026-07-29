import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
WORKSPACE_DIR = Path(__file__).parent.resolve()
DATA_DIR = WORKSPACE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# API Keys and Endpoints
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model Selection
# LLM variant generator model
LLM_MODEL = "meta/llama-3.1-8b-instruct"
# Speech-to-Text Model (ASR)
ASR_MODEL = "openai/whisper-large-v3"
# Text-to-Speech Model (TTS)
TTS_MODEL = "nvidia/magpie-tts-multilingual"

# Audio capture configurations
SAMPLE_RATE = 16000  # Standard sample rate for speech recognition
RECORDING_DURATION = 15  # Default recording duration in seconds
TEMP_INPUT_FILE = str(DATA_DIR / "temp_question.wav")
TEMP_OUTPUT_FILE = str(DATA_DIR / "temp_response.wav")

# TRIBE v2 Engagement Brain-Region Weights
# The sum of weights should ideally normalize to 1.0
WEIGHT_PFC = 0.3       # Prefrontal Cortex: Cognitive clarity, depth, logic
WEIGHT_AMYGDALA = 0.3  # Amygdala: Emotional resonance, intensity, valence
WEIGHT_TEMPORAL = 0.2  # Temporal Lobe: Auditory flow, rhythm, natural phonetics
WEIGHT_NACC = 0.2      # Nucleus Accumbens: Engagement hook, reward value

# Storage Configurations
DB_PATH = str(DATA_DIR / "voice_agent.db")
CSV_PATH = str(DATA_DIR / "interaction_logs.csv")
SAVED_AUDIO_DIR = DATA_DIR / "audio_records"
SAVED_AUDIO_DIR.mkdir(exist_ok=True)

# Toggle official model vs fast simulated fMRI pathway
# False is recommended for smooth, instant web presentations (under 2 seconds)
# True runs the heavy official neural model (~5 minutes on local CPU)
USE_OFFICIAL_MODEL = False
