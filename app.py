import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import sys
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import pipeline modules
import config
import speech_to_text
import generate_variants
import score_engagement
import rank_select
import text_to_speech
import logger

app = FastAPI(title="FinoLusion Cognitive CFO Dashboard")

# Ensure folders exist
data_dir = Path(config.WORKSPACE_DIR) / "data"
plots_dir = data_dir / "brain_plots"
reports_dir = Path(config.WORKSPACE_DIR) / "reports"

plots_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

# Mount static asset folders
app.mount("/data/brain_plots", StaticFiles(directory=str(plots_dir)), name="brain_plots")
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

@app.get("/data/temp_response.wav")
def get_latest_audio():
    wav_path = Path(config.WORKSPACE_DIR) / "data" / "temp_response.wav"
    if wav_path.exists():
        return FileResponse(str(wav_path), media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/record")
def record_and_transcribe():
    """
    Triggers microphone recording for 15 seconds, transcribes ASR.
    """
    wav_input = config.TEMP_INPUT_FILE
    record_success = speech_to_text.record_audio(wav_input, duration=config.RECORDING_DURATION)
    if not record_success:
        raise HTTPException(status_code=500, detail="Failed to capture microphone audio.")
        
    question = speech_to_text.transcribe_audio(wav_input)
    
    import re
    is_hindi = bool(re.search(r'[\u0900-\u097F]', question))
    
    return {
        "success": True,
        "question": question,
        "is_hindi": is_hindi
    }

@app.post("/api/submit")
def submit_question(req: QuestionRequest):
    """
    Submits a transcribed or typed question, runs batch fMRI predictions,
    synthesizes audio response, and writes SQLite logs and Word report.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    # Step 1: Generate 3 variants
    variants = generate_variants.generate_answer_variants(question)
    if not variants or len(variants) < 3:
        raise HTTPException(status_code=500, detail="Failed to generate 3 response variations.")
        
    # Step 2: Score in batch
    scored_variants = score_engagement.score_phrasings_batch(variants)
    variant_types = ["Analytical", "Strategic", "Operational"]
    for idx, v in enumerate(scored_variants):
        v["original_index"] = idx
        v["type"] = variant_types[idx]
    
    # Step 3: Rank and select
    selected_variant, ranked_variants = rank_select.rank_and_select(scored_variants)
    
    # Step 4: Synthesize TTS
    wav_output = config.TEMP_OUTPUT_FILE
    text_to_speech.text_to_speech(selected_variant["text"], wav_output, play=False)
    
    # Step 5: Log interaction & generate DOCX report
    archived_audio_path, report_path = logger.log_interaction(
        question,
        ranked_variants,
        selected_variant,
        wav_output
    )
    
    # Calculate min score for relative delta reference
    min_score = min(v["score"] for v in ranked_variants)
    
    processed_variants = []
    for v in ranked_variants:
        # Calculate relative delta
        score = v["score"]
        if min_score > 0 and score != min_score:
            rel_delta = ((score - min_score) / min_score) * 100
            rel_delta_str = f"+{rel_delta:.2f}%"
        else:
            rel_delta_str = "Baseline"
            
        plot_path = v.get("brain_plot_path")
        plot_url = f"/data/brain_plots/{os.path.basename(plot_path)}" if plot_path else None
        
        # Generate dynamic fMRI analysis for this variant
        acts = v["activations"]
        analysis = logger.generate_brain_analysis(
            v["text"],
            acts["PFC"],
            acts["Amygdala"],
            acts["Temporal"],
            acts["NAcc"]
        )
        
        processed_variants.append({
            "text": v["text"],
            "score": v["score"],
            "rel_delta": rel_delta_str,
            "activations": v["activations"],
            "brain_plot_url": plot_url,
            "analysis": analysis,
            "type": v.get("type", "Unknown"),
            "original_index": v.get("original_index", 0)
        })
        
    return {
        "success": True,
        "variants": processed_variants,
        "winner": {
            "text": selected_variant["text"],
            "score": selected_variant["score"],
            "type": selected_variant.get("type", "Unknown")
        },
        "report_filename": os.path.basename(report_path) if report_path else None
    }

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinoLusion | Cognitive CFO Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        :root {
            --bg-base: #0B0F19;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #10B981;
            --primary-glow: rgba(16, 185, 129, 0.2);
            --accent: #3B82F6;
            --text-main: #F3F4F6;
            --text-muted: #9CA3AF;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            background-color: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .logo {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.04);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            border: 1px solid var(--border-color);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--primary);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--primary);
        }

        main {
            flex: 1;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            padding: 2rem;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            transition: all 0.3s ease;
        }

        .input-card-wrapper {
            width: 100%;
            max-width: 500px;
            transition: all 0.3s ease;
        }

        main.results-active {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        @media (min-width: 1024px) {
            main.results-active {
                grid-template-columns: 480px 1fr;
            }
        }

        main.results-active .input-card-wrapper {
            max-width: 100%;
        }

        .card {
            background-color: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .toggle-container {
            display: flex;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.25rem;
        }

        .toggle-btn {
            flex: 1;
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 0.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.875rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .toggle-btn.active {
            background-color: var(--primary);
            color: white;
            box-shadow: 0 2px 8px var(--primary-glow);
        }

        .record-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem 0;
            gap: 1.5rem;
        }

        .mic-btn {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px var(--primary-glow);
            transition: all 0.3s;
            position: relative;
        }

        .mic-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.4);
        }

        .mic-btn.recording {
            animation: pulse 1.5s infinite;
            background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.4);
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.08); }
            100% { transform: scale(1); }
        }

        .wave-animation {
            display: none;
            align-items: center;
            gap: 4px;
            height: 30px;
        }

        .wave-bar {
            width: 4px;
            height: 8px;
            background-color: #EF4444;
            border-radius: 4px;
            animation: wave 1.2s ease-in-out infinite;
        }

        .wave-bar:nth-child(2) { animation-delay: 0.15s; }
        .wave-bar:nth-child(3) { animation-delay: 0.3s; }
        .wave-bar:nth-child(4) { animation-delay: 0.45s; }
        .wave-bar:nth-child(5) { animation-delay: 0.6s; }

        @keyframes wave {
            0%, 100% { height: 8px; }
            50% { height: 28px; }
        }

        .textarea {
            width: 100%;
            height: 120px;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            padding: 0.75rem;
            font-family: inherit;
            resize: none;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        .textarea:focus {
            border-color: var(--primary);
        }

        .btn {
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn:hover {
            background-color: #059669;
        }

        .btn-sec {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
        }

        .btn-sec:hover {
            background-color: rgba(255, 255, 255, 0.08);
        }

        .transcription-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .transcription-header {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent);
        }

        .leaderboard-row {
            display: grid;
            grid-template-columns: 50px 1fr auto;
            align-items: center;
            gap: 1rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            transition: border-color 0.2s;
        }

        .leaderboard-row.winner {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.04);
        }

        .rank-badge {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--text-muted);
        }

        .winner .rank-badge {
            color: var(--primary);
        }

        .phrasing-text {
            font-size: 0.875rem;
            line-height: 1.4;
        }

        .score-info {
            text-align: right;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .score-val {
            font-weight: 700;
            font-size: 0.95rem;
        }

        .delta-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            background-color: rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
        }

        .delta-badge.positive {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--primary);
        }

        .tabs-header {
            display: flex;
            border-bottom: 1px solid var(--border-color);
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 0.75rem 1.25rem;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.875rem;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }

        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }

        .tab-content {
            display: none;
            flex-direction: column;
            gap: 1.5rem;
        }

        .tab-content.active {
            display: flex;
        }

        .brain-display {
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            aspect-ratio: 16/9;
            position: relative;
        }

        .brain-img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .metric-bar-container {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .metric-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .progress-bar-bg {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            background-color: var(--accent);
            border-radius: 3px;
        }

        .metric-bar-container.pfc .progress-bar-fill { background-color: #3B82F6; }
        .metric-bar-container.amy .progress-bar-fill { background-color: #EC4899; }
        .metric-bar-container.temp .progress-bar-fill { background-color: #10B981; }
        .metric-bar-container.nacc .progress-bar-fill { background-color: #F59E0B; }

        .analysis-text {
            background-color: rgba(255, 255, 255, 0.02);
            border-left: 3px solid var(--primary);
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.875rem;
            line-height: 1.5;
            font-style: italic;
        }

        .audio-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .audio-player-controls {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
        }

        audio {
            width: 100%;
            height: 32px;
            outline: none;
        }

        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(11, 15, 25, 0.9);
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            z-index: 10;
            border-radius: 16px;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid var(--border-color);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <i data-lucide="brain-circuit"></i>
            FinoLusion CFO
        </div>
        <div class="status">
            <div class="status-dot"></div>
            Autonomous Twin Connected
        </div>
    </header>

    <main id="main-container">
        <!-- Input Card Wrapper -->
        <div class="input-card-wrapper">
            <div class="card" style="position: relative; width: 100%;">
                <h2><i data-lucide="message-square-plus"></i> Input Channel</h2>
                
                <div class="toggle-container">
                    <button class="toggle-btn active" id="btn-mode-voice" onclick="setMode('voice')">
                        <i data-lucide="mic"></i> Voice Mode
                    </button>
                    <button class="toggle-btn" id="btn-mode-text" onclick="setMode('text')">
                        <i data-lucide="keyboard"></i> Text Mode
                    </button>
                </div>

                <!-- Voice Record Display -->
                <div id="voice-input-container" class="record-section">
                    <button class="mic-btn" id="btn-record" onclick="startRecording()">
                        <i data-lucide="mic" style="width: 36px; height: 36px;"></i>
                    </button>
                    <div class="wave-animation" id="wave-animation">
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                    </div>
                    <div id="countdown-timer" style="font-size: 1.5rem; font-weight: 700; color: #EF4444; display: none; font-family: 'Outfit', sans-serif;">15s</div>
                    <p id="record-status" style="font-size: 0.875rem; color: var(--text-muted);">
                        Click the microphone to start recording (15 seconds)
                    </p>
                </div>

                <!-- Text Entry Display -->
                <div id="text-input-container" style="display: none; flex-direction: column; gap: 1rem;">
                    <textarea class="textarea" id="text-question" placeholder="Ask FinoLusion your corporate finance or business query..."></textarea>
                    <button class="btn" onclick="submitTextQuestion()">
                        <i data-lucide="send"></i> Submit Query
                    </button>
                </div>

                <!-- Preview Card -->
                <div id="preview-card" class="transcription-card" style="display: none;">
                    <div class="transcription-header" id="preview-lang-badge">Transcribed Question</div>
                    <p id="preview-question" style="font-size: 0.95rem; font-weight: 500;"></p>

                    <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                        <button class="btn btn-sec" style="flex: 1;" onclick="resetCapture()">
                            <i data-lucide="rotate-ccw"></i> Retake
                        </button>
                        <button class="btn" style="flex: 1;" onclick="processFinalQuestion()">
                            <i data-lucide="play"></i> Analyze Engagement
                        </button>
                    </div>
                </div>

                <div class="loading-overlay" id="left-loading">
                    <div class="spinner"></div>
                    <p id="loading-msg" style="font-weight: 600;">Processing speech...</p>
                </div>
            </div>
        </div>

        <!-- RIGHT PANEL: DASHBOARD RESULTS -->
        <div class="card" id="results-panel" style="display: none; position: relative; min-height: 550px;">
            <!-- Loading State View -->
            <div id="results-loading-state" style="display: flex; flex-direction: column; align-items: center; justify-content: center; flex: 1; gap: 1.5rem; padding: 3rem 0; width: 100%;">
                <div class="spinner" style="width: 50px; height: 50px;"></div>
                <div style="text-align: center;">
                    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.15rem; margin-bottom: 0.5rem;">Running Cognitive fMRI Simulation</h3>
                    <p style="font-size: 0.875rem; color: var(--text-muted); max-width: 340px; margin: 0 auto; line-height: 1.4;">
                        Predicting cortical activations for the stakeholder variants. This takes about 90-120 seconds on CPU.
                    </p>
                </div>
                
                <!-- Status Timeline Checkbox List -->
                <div style="display: flex; flex-direction: column; gap: 0.75rem; width: 100%; max-width: 320px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-color); padding: 1rem; border-radius: 8px; font-size: 0.875rem;">
                    <div id="step-llm" style="display: flex; align-items: center; justify-content: space-between; color: var(--text-muted);">
                        <span>Generating stakeholder phrasings</span>
                        <span class="step-status"><i data-lucide="circle-dashed" style="width: 16px; height: 16px;"></i></span>
                    </div>
                    <div id="step-fmri" style="display: flex; align-items: center; justify-content: space-between; color: var(--text-muted);">
                        <span>Running TRIBE v2 fMRI model</span>
                        <span class="step-status"><i data-lucide="circle-dashed" style="width: 16px; height: 16px;"></i></span>
                    </div>
                    <div id="step-plots" style="display: flex; align-items: center; justify-content: space-between; color: var(--text-muted);">
                        <span>Rendering 3D cortical plots</span>
                        <span class="step-status"><i data-lucide="circle-dashed" style="width: 16px; height: 16px;"></i></span>
                    </div>
                    <div id="step-tts" style="display: flex; align-items: center; justify-content: space-between; color: var(--text-muted);">
                        <span>Synthesizing cloud TTS voice</span>
                        <span class="step-status"><i data-lucide="circle-dashed" style="width: 16px; height: 16px;"></i></span>
                    </div>
                </div>
            </div>

            <!-- Content State View (Shown after API completes) -->
            <div id="results-content-state" style="display: none; flex-direction: column; gap: 1.5rem; width: 100%;">
                <h2><i data-lucide="bar-chart-3"></i> Cognitive Analysis Dashboard</h2>

                <!-- Leaderboard -->
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div class="transcription-header">TRIBE v2 Engagement Leaderboard</div>
                    <div id="leaderboard-rows" style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <!-- Filled Dynamically -->
                    </div>
                </div>

                <!-- Brain Plot Visualizations -->
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div class="tabs-header" id="tabs-header">
                        <!-- Filled Dynamically -->
                    </div>

                    <div id="tabs-content-container">
                        <!-- Filled Dynamically -->
                    </div>
                </div>

                <!-- Audio Response Player -->
                <div class="audio-card">
                    <div class="audio-player-controls">
                        <i data-lucide="volume-2" style="color: var(--primary);"></i>
                        <audio id="audio-response-player" controls></audio>
                    </div>
                </div>

                <!-- Document Download -->
                <button class="btn btn-sec" id="btn-download-report" onclick="downloadReport()">
                    <i data-lucide="file-text"></i> Export Word (.docx) Report
                </button>
            </div>
        </div>
    </main>

    <script>
        let currentMode = 'voice';
        let capturedQuestion = '';
        let reportFilename = '';
        let timerInterval = null;

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('btn-mode-voice').classList.toggle('active', mode === 'voice');
            document.getElementById('btn-mode-text').classList.toggle('active', mode === 'text');
            
            document.getElementById('voice-input-container').style.display = mode === 'voice' ? 'flex' : 'none';
            document.getElementById('text-input-container').style.display = mode === 'text' ? 'flex' : 'none';
            document.getElementById('preview-card').style.display = 'none';
        }

        function startRecording() {
            const micBtn = document.getElementById('btn-record');
            if (micBtn.classList.contains('recording')) {
                return; // Prevent duplicate concurrent runs
            }
            
            const waveAnim = document.getElementById('wave-animation');
            const statusText = document.getElementById('record-status');
            const timerEl = document.getElementById('countdown-timer');
            const previewCard = document.getElementById('preview-card');

            // Reset UI states
            previewCard.style.display = 'none';
            micBtn.classList.add('recording');
            waveAnim.style.display = 'flex';
            timerEl.style.display = 'block';
            
            if (timerInterval) {
                clearInterval(timerInterval);
            }
            
            let timeLeft = 15;
            timerEl.innerText = `${timeLeft}s`;
            statusText.innerText = 'Microphone active. Speak now...';
            
            timerInterval = setInterval(() => {
                timeLeft -= 1;
                if (timeLeft > 0) {
                    timerEl.innerText = `${timeLeft}s`;
                } else {
                    clearInterval(timerInterval);
                    timerEl.innerText = 'Transcribing...';
                    statusText.innerText = 'Processing your speech...';
                }
            }, 1000);

            // Hit record API
            fetch('/api/record', { method: 'POST' })
                .then(res => {
                    clearInterval(timerInterval);
                    if (!res.ok) throw new Error("Mic recording failed.");
                    return res.json();
                })
                .then(data => {
                    micBtn.classList.remove('recording');
                    waveAnim.style.display = 'none';
                    timerEl.style.display = 'none';
                    statusText.innerText = 'Click the microphone to start recording (15 seconds)';
                    
                    capturedQuestion = data.question;
                    document.getElementById('preview-question').innerText = data.question;
                    
                    if (data.is_hindi) {
                        document.getElementById('preview-lang-badge').innerText = "Transcribed Question (Hindi)";
                    } else {
                        document.getElementById('preview-lang-badge').innerText = "Transcribed Question (English)";
                    }
                    
                    document.getElementById('preview-card').style.display = 'flex';
                })
                .catch(err => {
                    clearInterval(timerInterval);
                    micBtn.classList.remove('recording');
                    waveAnim.style.display = 'none';
                    timerEl.style.display = 'none';
                    statusText.innerText = 'Click the microphone to start recording (15 seconds)';
                    alert("Recording failed. Make sure a microphone is connected.");
                });
        }

        function resetCapture() {
            document.getElementById('preview-card').style.display = 'none';
            capturedQuestion = '';
        }

        function submitTextQuestion() {
            const qText = document.getElementById('text-question').value.trim();
            if (!qText) {
                alert("Please enter a question.");
                return;
            }
            capturedQuestion = qText;
            processFinalQuestion();
        }

        function updateStepStatus(stepId, status) {
            const stepEl = document.getElementById(stepId);
            const statusSpan = stepEl.querySelector('.step-status');
            
            if (status === 'pending') {
                stepEl.style.color = 'var(--text-muted)';
                statusSpan.innerHTML = '<i data-lucide="circle-dashed" style="width: 16px; height: 16px;"></i>';
            } else if (status === 'running') {
                stepEl.style.color = 'var(--accent)';
                statusSpan.innerHTML = '<i data-lucide="loader" class="spinner" style="width: 16px; height: 16px; border: 2px solid transparent; border-top-color: var(--accent); animation: spin 1s linear infinite; border-radius: 50%;"></i>';
            } else if (status === 'success') {
                stepEl.style.color = 'var(--primary)';
                statusSpan.innerHTML = '<i data-lucide="check-circle" style="width: 16px; height: 16px; color: var(--primary);"></i>';
            }
            lucide.createIcons();
        }

        function processFinalQuestion() {
            const resultsPanel = document.getElementById('results-panel');
            const mainContainer = document.getElementById('main-container');
            const loadState = document.getElementById('results-loading-state');
            const contentState = document.getElementById('results-content-state');
            
            // Show loading view initially
            resultsPanel.style.display = 'flex';
            loadState.style.display = 'flex';
            contentState.style.display = 'none';
            mainContainer.classList.add('results-active');

            // Reset checklist
            updateStepStatus('step-llm', 'pending');
            updateStepStatus('step-fmri', 'pending');
            updateStepStatus('step-plots', 'pending');
            updateStepStatus('step-tts', 'pending');

            // Step 1: LLM Generation starts immediately
            updateStepStatus('step-llm', 'running');

            // Simulate step transition (LLM takes about 3 seconds)
            const llmTimer = setTimeout(() => {
                updateStepStatus('step-llm', 'success');
                updateStepStatus('step-fmri', 'running');
            }, 3000);

            fetch('/api/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: capturedQuestion })
            })
            .then(res => {
                clearTimeout(llmTimer);
                if (!res.ok) throw new Error("Brain prediction pipeline failed.");
                return res.json();
            })
            .then(data => {
                // Complete all steps in the list
                updateStepStatus('step-llm', 'success');
                updateStepStatus('step-fmri', 'success');
                updateStepStatus('step-plots', 'success');
                updateStepStatus('step-tts', 'success');
                
                // Transition view
                setTimeout(() => {
                    loadState.style.display = 'none';
                    contentState.style.display = 'flex';
                }, 500);

                reportFilename = data.report_filename;
                
                // 1. Fill Leaderboard
                const rowsContainer = document.getElementById('leaderboard-rows');
                rowsContainer.innerHTML = '';
                data.variants.forEach((v, idx) => {
                    const row = document.createElement('div');
                    row.className = `leaderboard-row ${idx === 0 ? 'winner' : ''}`;
                    
                    const isWinner = idx === 0;
                    const deltaClass = v.rel_delta !== 'Baseline' ? 'positive' : '';
                    
                    row.innerHTML = `
                        <div class="rank-badge">${isWinner ? '★' : `#${idx+1}`}</div>
                        <div class="phrasing-text">"${v.text}"</div>
                        <div class="score-info">
                            <div class="score-val">${v.score.toFixed(4)}</div>
                            <div class="delta-badge ${deltaClass}">${v.rel_delta}</div>
                        </div>
                    `;
                    rowsContainer.appendChild(row);
                });
                
                // 2. Fill Tabs and Brain Plots
                const tabsHeader = document.getElementById('tabs-header');
                const tabsContainer = document.getElementById('tabs-content-container');
                tabsHeader.innerHTML = '';
                tabsContainer.innerHTML = '';
                
                data.variants.forEach((v, idx) => {
                    // Header tab button
                    const tabBtn = document.createElement('button');
                    tabBtn.className = `tab-btn ${idx === 0 ? 'active' : ''}`;
                    tabBtn.innerText = idx === 0 ? '★ Winner' : `Variant ${idx+1}`;
                    tabBtn.onclick = () => switchTab(idx);
                    tabsHeader.appendChild(tabBtn);
                    
                    // Content
                    const tabContent = document.createElement('div');
                    tabContent.className = `tab-content ${idx === 0 ? 'active' : ''}`;
                    tabContent.id = `tab-content-${idx}`;
                    
                    const pfcPct = Math.round(v.activations.PFC * 100);
                    const amyPct = Math.round(v.activations.Amygdala * 100);
                    const tempPct = Math.round(v.activations.Temporal * 100);
                    const naccPct = Math.round(v.activations.NAcc * 100);
                    
                    const plotHtml = v.brain_plot_url 
                        ? `<img src="${v.brain_plot_url}" class="brain-img" alt="Brain Activations">`
                        : `<div style="color: var(--text-muted); font-size: 0.875rem;">3D Brain Plot Unavailable</div>`;
                        
                    const analysisText = v.analysis;
                    
                    tabContent.innerHTML = `
                        <div class="brain-display">
                            ${plotHtml}
                        </div>
                        <div class="metrics-grid">
                            <div class="metric-bar-container pfc">
                                <div class="metric-label-row"><span>Prefrontal Cortex (Logic)</span><span>${pfcPct}%</span></div>
                                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${pfcPct}%"></div></div>
                            </div>
                            <div class="metric-bar-container amy">
                                <div class="metric-label-row"><span>Amygdala (Excitement)</span><span>${amyPct}%</span></div>
                                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${amyPct}%"></div></div>
                            </div>
                            <div class="metric-bar-container temp">
                                <div class="metric-label-row"><span>Temporal (Cadence)</span><span>${tempPct}%</span></div>
                                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${tempPct}%"></div></div>
                            </div>
                            <div class="metric-bar-container nacc">
                                <div class="metric-label-row"><span>Nucleus Accumbens (Reward)</span><span>${naccPct}%</span></div>
                                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${naccPct}%"></div></div>
                            </div>
                        </div>
                        <div class="analysis-text">
                            ${analysisText}
                        </div>
                    `;
                    tabsContainer.appendChild(tabContent);
                });
                
                // 3. Set Audio Player
                const audioPlayer = document.getElementById('audio-response-player');
                audioPlayer.src = `/data/temp_response.wav?t=${Date.now()}`;
                audioPlayer.load();
                audioPlayer.play().catch(e => console.log("Auto-playback blocked by browser policy. User action required."));
                
                lucide.createIcons();
            })
            .catch(err => {
                clearTimeout(llmTimer);
                loadState.style.display = 'none';
                alert("Brain simulation run failed. Check console error messages.");
            });
        }

        function switchTab(idx) {
            const btns = document.querySelectorAll('.tab-btn');
            const contents = document.querySelectorAll('.tab-content');
            
            btns.forEach((btn, i) => btn.classList.toggle('active', i === idx));
            contents.forEach((content, i) => content.classList.toggle('active', i === idx));
        }

        function downloadReport() {
            if (!reportFilename) {
                alert("No report generated yet.");
                return;
            }
            window.open(`/reports/${reportFilename}`);
        }

        // Initialize icons on page load
        document.addEventListener('DOMContentLoaded', () => {
            lucide.createIcons();
        });
    </script>
</body>
</html>"""
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
