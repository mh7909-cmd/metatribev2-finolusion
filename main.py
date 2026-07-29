import argparse
import sys
import os
from datetime import datetime

import config
import speech_to_text
import generate_variants
import score_engagement
import rank_select
import text_to_speech
import logger

# ANSI color codes for premium console output style
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner():
    banner = f"""
{BLUE}{BOLD}================================================================================
          TRIBE v2-INSPIRED VOICE AGENT ENGAGEMENT PIPELINE
================================================================================{RESET}
    """
    print(banner)

def main():
    parser = argparse.ArgumentParser(
        description="Run the Voice Agent Pipeline with TRIBE v2-inspired engagement selection."
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Skip microphone input and type the question directly in the console."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=config.RECORDING_DURATION,
        help=f"Microphone recording duration in seconds (default: {config.RECORDING_DURATION})."
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Synthesize TTS but skip playing the audio output out loud."
    )
    args = parser.parse_args()

    print_banner()
    
    # Verify environment key warning
    if not config.NVIDIA_API_KEY:
        print(f"{YELLOW}[System Warning] NVIDIA_API_KEY is not set in environment or .env file.")
        print(f"The pipeline will run using local heuristics and mock generators.{RESET}\n")
    
    # -------------------------------------------------------------------------
    # STEP 1: Speech-To-Text (ASR)
    # -------------------------------------------------------------------------
    print(f"{BOLD}{CYAN}[Step 1/6] Input Selection & Capture{RESET}")
    
    is_text_mode = args.text
    if not is_text_mode:
        print(f"\n{BOLD}Select Input Mode:{RESET}")
        print("  [1] Speak via Microphone (Voice Mode)")
        print("  [2] Type question in Console (Text Mode)")
        try:
            choice = input(f"\n{CYAN}Choice [1/2, default 1]: {RESET}").strip()
            if choice == "2":
                is_text_mode = True
        except (KeyboardInterrupt, EOFError):
            print("\n[System] Cancelled by user.")
            sys.exit(0)

    question = ""
    if is_text_mode:
        print("\n[ASR] Running in text-only mode.")
        question = speech_to_text.get_text_input_fallback()
    else:
        wav_input = config.TEMP_INPUT_FILE
        while True:
            record_success = speech_to_text.record_audio(wav_input, duration=args.duration)
            if record_success:
                question = speech_to_text.transcribe_audio(wav_input)
            else:
                print(f"{YELLOW}[ASR] Failed to capture audio. Falling back to console text entry...{RESET}")
                question = speech_to_text.get_text_input_fallback()
                break
                
            print(f"\n{GREEN}{BOLD}>>> Transcribed Question:{RESET} {BOLD}\"{question}\"{RESET}")
            
            # Fetch translation dynamically
            translation = speech_to_text.translate_question(question)
            if translation:
                import re
                is_hindi = bool(re.search(r'[\u0900-\u097F]', question))
                label = "English Translation" if is_hindi else "Hindi Translation"
                print(f"{CYAN}{BOLD}>>> {label}:{RESET} {BOLD}\"{translation}\"{RESET}")
                
            try:
                confirm = input(f"{CYAN}Is this correct? [Y/n, default Y]: {RESET}").strip().lower()
                if confirm in ["", "y", "yes"]:
                    break
                else:
                    print(f"\n{YELLOW}[ASR] Let's try again! Prepare to speak...{RESET}")
            except (KeyboardInterrupt, EOFError):
                print("\n[System] Cancelled by user.")
                sys.exit(0)
                
    print()

    # -------------------------------------------------------------------------
    # STEP 2: Answer Generation (LLM Variants)
    # -------------------------------------------------------------------------
    print(f"{BOLD}{CYAN}[Step 2/6] Answer Generation (3 Variants){RESET}")
    variants = generate_variants.generate_answer_variants(question)
    
    if not variants or len(variants) < 3:
        print(f"{YELLOW}[Generator] Error: Could not generate 3 valid variants. Terminating.{RESET}")
        sys.exit(1)
        
    print()

    # -------------------------------------------------------------------------
    # STEP 3: Engagement Scoring (TRIBE v2-inspired Brain Encoding)
    # -------------------------------------------------------------------------
    print(f"{BOLD}{CYAN}[Step 3/6] Engagement Scoring Module (TRIBE v2 Neural Twin){RESET}")
    print(f"\n{BOLD}--- Running fMRI Neural Twin Batch Predictions ---{RESET}")
    scored_variants = score_engagement.score_phrasings_batch(variants)

    print()

    # -------------------------------------------------------------------------
    # STEP 4: Ranking and Selection
    # -------------------------------------------------------------------------
    print(f"{BOLD}{CYAN}[Step 4/6] Ranking & Variant Selection{RESET}")
    selected_variant, ranked_variants = rank_select.rank_and_select(scored_variants)
    
    winning_text = selected_variant["text"]
    winning_score = selected_variant["score"]

    # -------------------------------------------------------------------------
    # STEP 5: Text-To-Speech Synthesis & Playback
    # -------------------------------------------------------------------------
    print(f"{BOLD}{CYAN}[Step 5/6] Text-To-Speech Synthesis{RESET}")
    wav_output = config.TEMP_OUTPUT_FILE
    
    # Synthesize the audio
    tts_success = text_to_speech.text_to_speech(winning_text, wav_output)
    
    if not tts_success:
        print(f"{YELLOW}[TTS] Synthesis or local fallback speaking encountered an error.{RESET}")
    
    if args.no_play:
        print("[TTS] Playback skipped due to --no-play flag.")
        
    print()

    # -------------------------------------------------------------------------
    # STEP 6: Logging and Storage
    # -------------------------------------------------------------------------
    print(f"{BOLD}{CYAN}[Step 6/6] Logging and Database Storage{RESET}")
    
    # Store everything and persistent-archive the output wav file
    final_audio_path, report_path = logger.log_interaction(
        question=question,
        variants_with_scores=scored_variants,
        selected_variant=selected_variant,
        temp_audio_path=wav_output if tts_success else ""
    )
    
    print(f"\n{GREEN}{BOLD}================================================================================")
    print("                      PIPELINE EXECUTION SUCCESSFULLY COMPLETED                 ")
    print(f"================================================================================")
    print(f"  • Question: \"{question}\"")
    print(f"  • Selected Answer (Score: {winning_score:.4f}):")
    print(f"    \"{winning_text}\"")
    print(f"  • SQLite Database: {config.DB_PATH}")
    print(f"  • CSV Export File: {config.CSV_PATH}")
    if final_audio_path:
        print(f"  • Archived Audio: {final_audio_path}")
    
    winning_plot = selected_variant.get("brain_plot_path")
    if winning_plot:
        print(f"  • Brain Simulation Plot: {winning_plot}")
    if report_path:
        print(f"  • Interactive HTML/MD Report: {report_path}")
    print(f"================================================================================")
    print(f"{RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[System] Voice pipeline stopped by user.")
        sys.exit(0)
