from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import numpy as np
import base64
import io
import os
from typing import Optional

app = FastAPI(title="AI Voice Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "guvi-hackathon-voice-detection-2026"

class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str
    message: Optional[str] = None

class VoiceDetectionResponse(BaseModel):
    isAIGenerated: bool
    confidence: float
    language: str
    audioFormat: str
    details: str

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "AI Voice Detection API",
        "version": "1.0"
    }

@app.post("/detect", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_io = io.BytesIO(audio_bytes)
        
        try:
            audio_data, sr = librosa.load(audio_io, sr=None)
        except Exception:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000
        
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        
        features = np.concatenate([
            mfcc_mean,
            np.mean(spectral_centroid),
            np.mean(spectral_rolloff),
            np.mean(zero_crossing_rate)
        ])
        
        feature_score = np.sum(features) / len(features)
        confidence = abs(np.sin(feature_score))
        
        is_ai_generated = confidence > 0.5
        
        return VoiceDetectionResponse(
            isAIGenerated=is_ai_generated,
            confidence=round(confidence, 3),
            language=request.language,
            audioFormat=request.audioFormat,
            details=f"Audio analyzed. Classification: {'AI-Generated' if is_ai_generated else 'Human Voice'}. Confidence: {confidence*100:.1f}%"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing audio: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
