import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import subprocess
import sys
from openai import OpenAI
import config

def split_into_speech_chunks(text: str, max_chars: int = 300) -> list[str]:
    """
    Splits text into chunks of maximum max_chars length, keeping sentence boundaries intact.
    Splits on periods (.), Hindi full stops (।), question marks (?), exclamation marks (!), or commas (,).
    """
    import re
    sentences = re.split(r'([.।?!,\n]+)', text)
    
    chunks = []
    current_chunk = ""
    
    for part in sentences:
        if not part:
            continue
        if len(current_chunk) + len(part) < max_chars:
            current_chunk += part
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = part
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def text_to_speech(text: str, output_filepath: str, play: bool = True) -> bool:
    """
    Synthesizes text into speech.
    Attempts to call NVIDIA's hosted Magpie-TTS gRPC API using your API key.
    Falls back cleanly to macOS native 'say' synthesizer if the API key is not present or a network error occurs.
    """
    succeeded = False
    
    if config.NVIDIA_API_KEY:
        try:
            print("[TTS] Synthesizing speech via NVIDIA Magpie-TTS Cloud (gRPC)...")
            import riva.client
            from riva.client.proto.riva_audio_pb2 import AudioEncoding
            import wave
            
            auth = riva.client.Auth(
                uri="grpc.nvcf.nvidia.com:443",
                use_ssl=True,
                metadata_args=[
                    ["function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"],
                    ["authorization", f"Bearer {config.NVIDIA_API_KEY}"]
                ]
            )
            
            import re
            is_hindi = bool(re.search(r'[\u0900-\u097F]', text))
            voice_name = "Magpie-Multilingual.EN-US.Aria.Calm"
            language_code = "en-US"
            
            if is_hindi:
                voice_name = "Magpie-Multilingual.HI-IN.Aria.Calm"
                language_code = "hi-IN"
                print(f"[TTS] Detected Hindi text. Using voice: {voice_name}")
            
            service = riva.client.SpeechSynthesisService(auth)
            sample_rate_hz = 22050
            
            # Triton has a maximum sequence length limit (400 characters).
            # We split the text into chunks under 300 characters.
            chunks = split_into_speech_chunks(text, max_chars=300)
            
            combined_audio = b""
            for chunk in chunks:
                if not chunk.strip():
                    continue
                resp = service.synthesize(
                    text=chunk,
                    voice_name=voice_name,
                    language_code=language_code,
                    sample_rate_hz=sample_rate_hz,
                    encoding=AudioEncoding.LINEAR_PCM
                )
                combined_audio += resp.audio
                
            with wave.open(output_filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate_hz)
                wf.writeframesraw(combined_audio)
                
            print(f"[TTS] Synthesized successfully via NVIDIA cloud ({len(chunks)} chunks). Audio saved to {output_filepath}")
            succeeded = True
        except Exception as e:
            print(f"[TTS] NVIDIA cloud synthesis failed: {e}. Switching to macOS fallback...")
            succeeded = run_local_tts_fallback(text, output_filepath)
    else:
        print("[TTS] No NVIDIA API key set. Using macOS fallback...")
        succeeded = run_local_tts_fallback(text, output_filepath)
        
    if succeeded:
        if play:
            # Play the synthesized audio (only for CLI/offline mode)
            play_audio(output_filepath)
        return True
    
    return False
 
def run_local_tts_fallback(text: str, output_filepath: str) -> bool:
    """
    Synthesizes speech using the macOS native 'say' command.
    Generates an audio file.
    """
    try:
        print("[TTS Fallback] Running macOS 'say' command to generate audio...")
        import re
        is_hindi = bool(re.search(r'[\u0900-\u097F]', text))
        
        cmd = ["say", "-o", output_filepath, "--data-format=LEI16@16000"]
        if is_hindi:
            try:
                # Check if Hindi voice Lekha is installed on the system
                voices_check = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
                if "Lekha" in voices_check.stdout:
                    cmd.extend(["-v", "Lekha"])
                    print("[TTS Fallback] Detected Lekha (macOS Hindi Voice). Using Lekha...")
            except Exception:
                pass
                
        cmd.append(text)
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"[TTS Fallback] Native audio generated at {output_filepath}")
        return True
    except Exception as e:
        print(f"[TTS Fallback] macOS 'say' export failed: {e}")
        # Final emergency fallback: speak out loud without file
        try:
            is_hindi = bool(re.search(r'[\u0900-\u097F]', text))
            cmd_say = ["say"]
            if is_hindi:
                try:
                    voices_check = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
                    if "Lekha" in voices_check.stdout:
                        cmd_say.extend(["-v", "Lekha"])
                except Exception:
                    pass
            cmd_say.append(text)
            subprocess.run(cmd_say)
            return True
        except Exception:
            return False

def play_audio(filepath: str):
    """
    Plays an audio file on macOS using the native 'afplay' CLI utility.
    """
    try:
        print(f"[Audio Player] Playing back audio: {filepath}...")
        # Run afplay in the background or foreground?
        # Foreground is better so main.py waits for speech to finish.
        subprocess.run(["afplay", filepath], check=True)
        print("[Audio Player] Playback completed.")
    except Exception as e:
        print(f"[Audio Player] Failed to play audio using afplay: {e}")
        # Windows / Linux compatibility safety block
        if sys.platform == "win32":
            import os
            os.system(f"start {filepath}")
        elif sys.platform == "linux":
            subprocess.run(["aplay", filepath])

if __name__ == "__main__":
    # Test TTS locally
    test_text = "The capital of France is Paris, which is its largest city."
    out_file = config.TEMP_OUTPUT_FILE
    text_to_speech(test_text, out_file)
