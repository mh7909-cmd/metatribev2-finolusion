import os
import time
import sounddevice as sd
import numpy as np
import soundfile as sf
from openai import OpenAI
import config

def record_audio(output_filepath: str, duration: int = config.RECORDING_DURATION, samplerate: int = config.SAMPLE_RATE) -> bool:
    """
    Records audio from the default microphone and saves it to a WAV file.
    Returns True if recording succeeded, False otherwise.
    """
    try:
        print(f"\n[STT] Preparing microphone to record for {duration} seconds...")
        # Get default input device info to make sure a microphone is available
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        if not default_input:
            print("[STT] Error: No input audio device (microphone) found.")
            return False

        # Play alert beep directly through sounddevice
        print("[STT] Beep! Get ready to speak...")
        try:
            fs_beep = 44100
            duration_beep = 0.3
            freq_beep = 1000
            t_beep = np.linspace(0, duration_beep, int(fs_beep * duration_beep), endpoint=False)
            beep_wave = np.sin(2 * np.pi * freq_beep * t_beep) * 0.3  # Play at comfortable 30% volume
            sd.play(beep_wave, samplerate=fs_beep)
            sd.wait()
        except Exception as beep_err:
            print(f"[STT] System beep playback failed, using fallback speech notification: {beep_err}")
            os.system('say "Go"')
        time.sleep(0.2)

        print("[STT] Recording started! Speak now...")
        # Record into numpy array
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        
        # Display progress countdown
        for remaining in range(duration, 0, -1):
            print(f"  --> {remaining} seconds remaining...", end="\r", flush=True)
            time.sleep(1)
        print("  --> Recording complete!                ")
        
        sd.wait()  # Wait until the recording is finished
        
        # Write to WAV file
        sf.write(output_filepath, recording, samplerate)
        print(f"[STT] Audio saved locally to {output_filepath}")
        return True
    except Exception as e:
        print(f"[STT] Audio recording failed: {e}")
        return False

def transcribe_audio(file_path: str) -> str:
    """
    Sends the audio file to Google Speech Recognition via speech_recognition.
    Dynamically supports both English (en-US) and Hindi (hi-IN) speech input.
    """
    try:
        import speech_recognition as sr
        import re
        print(f"[STT] Transcribing audio {file_path} using SpeechRecognition...")
        r = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio_data = r.record(source)
            
        # 1. Try English transcription
        try:
            text_en = r.recognize_google(audio_data, language="en-US").strip()
        except Exception:
            text_en = ""
            
        # 2. Try Hindi transcription
        try:
            text_hi = r.recognize_google(audio_data, language="hi-IN").strip()
        except Exception:
            text_hi = ""
            
        # Automatic language decision logic
        if text_en and text_hi:
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url=config.NVIDIA_BASE_URL,
                    api_key=config.NVIDIA_API_KEY
                )
                prompt = (
                    "You are an expert linguistics classifier. Analyze these two transcriptions of the same spoken audio:\n"
                    f"Candidate A (English ASR model): \"{text_en}\"\n"
                    f"Candidate B (Hindi ASR model): \"{text_hi}\"\n\n"
                    "Determine which language the speaker actually spoke.\n"
                    "Choose 'English' if they spoke English (even with an Indian accent, or talking about manufacturing/Nike/costs).\n"
                    "Choose 'Hindi' if they spoke Hindi or mixed Hinglish (e.g. 'trend kya chal raha hai').\n"
                    "Respond with ONLY the word 'English' or 'Hindi'."
                )
                response = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                decision = response.choices[0].message.content.strip().lower()
                if "english" in decision:
                    text = text_en
                    print(f"[STT] Llama decided English was spoken: \"{text}\"")
                else:
                    text = text_hi
                    print(f"[STT] Llama decided Hindi was spoken: \"{text}\"")
            except Exception as e:
                # Fallback to devanagari detection
                has_devanagari = bool(re.search(r'[\u0900-\u097F]', text_hi))
                if has_devanagari:
                    text = text_hi
                else:
                    text = text_en or text_hi
        elif text_en:
            text = text_en
            print(f"[STT] Detected English speech: \"{text}\"")
        elif text_hi:
            text = text_hi
            print(f"[STT] Detected Hindi speech: \"{text}\"")
        else:
            raise sr.UnknownValueError("Could not transcribe audio in English or Hindi.")
            
        formatted_text = format_transcription(text.strip())
        return formatted_text if formatted_text else text.strip()
    except Exception as e:
        print(f"[STT] SpeechRecognition failed with error: {e}. Falling back to manual text input.")
        return get_text_input_fallback()

def format_transcription(text: str) -> str:
    """
    Cleans up transcription grammar, casing, capitalization (e.g. Artificial General Intelligence),
    and punctuation (adds question marks/exclamation points) using Llama 3.1.
    """
    if not text.strip() or not config.NVIDIA_API_KEY:
        return text
    from openai import OpenAI
    try:
        client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY
        )
        prompt = (
            "You are a professional editor. Correct the capitalization, punctuation, and grammar of this transcribed speech text. "
            "Ensure proper nouns and acronyms are capitalized (e.g., capitalize 'Artificial General Intelligence' or 'Nike Air Force'). "
            "Add appropriate punctuation such as question marks or exclamation marks where they fit. "
            "Output ONLY the corrected text. Do not add any notes, explanations, or quotes around it.\n\n"
            f"Text: \"{text}\""
        )
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        return text

def translate_question(text: str) -> str:
    """
    Translates the question between English and Hindi using the NVIDIA Llama model.
    """
    if not config.NVIDIA_API_KEY:
        return ""
    import re
    from openai import OpenAI
    
    is_hindi = bool(re.search(r'[\u0900-\u097F]', text))
    source_lang = "Hindi" if is_hindi else "English"
    target_lang = "English" if is_hindi else "Hindi"
    
    try:
        client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY
        )
        prompt = (
            f"You are a professional translator. Translate the following sentence from {source_lang} to {target_lang}. "
            f"Provide only the direct translation. Do not add any conversational text, explanations, or notes.\n\n"
            f"Sentence: \"{text}\""
        )
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        translation = response.choices[0].message.content.strip().replace('"', '')
        return translation
    except Exception as e:
        return ""

def get_text_input_fallback() -> str:
    """
    Console input fallback to ensure the pipeline is testable even without mic/API.
    """
    print("\n[STT Fallback] Please type your question manually below:")
    user_input = input("Question: ").strip()
    while not user_input:
        user_input = input("Please type a valid question: ").strip()
    return user_input

if __name__ == "__main__":
    # Test STT locally
    wav_file = config.TEMP_INPUT_FILE
    if record_audio(wav_file, duration=3):
        text = transcribe_audio(wav_file)
        print(f"Result: {text}")
    else:
        text = get_text_input_fallback()
        print(f"Fallback Result: {text}")
