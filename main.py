import base64
from io import BytesIO
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import functions from convert.py
from convert import load_models, convert_audio_buffer

app = FastAPI()

# Load models only once on startup to avoid reloading on every request
hps, net_g, cmodel, smodel = load_models("configs/freevc.json", "checkpoints/freevc.pth")


class VoiceConversionRequest(BaseModel):
    source_base64: str
    target_base64: str


class BatchVoiceConversionRequest(BaseModel):
    samples: List[VoiceConversionRequest]


class VoiceConversionResponse(BaseModel):
    output_base64: str


@app.post("/convert-voices", response_model=List[VoiceConversionResponse])
async def convert_voices(batch_request: BatchVoiceConversionRequest):
    responses = []
    for idx, sample in enumerate(batch_request.samples):
        try:
            # Decode Base64 strings into bytes and wrap them in BytesIO objects
            source_buffer = BytesIO(base64.b64decode(sample.source_base64))
            target_buffer = BytesIO(base64.b64decode(sample.target_base64))
            
            # Use a title to identify the conversion
            title = f"sample_{idx}"
            
            # Convert the audio entirely in-memory
            output_buffer = convert_audio_buffer(source_buffer, target_buffer, title,
                                                   hps, net_g, cmodel, smodel)
            
            # Read output audio bytes and encode to Base64 for the response
            output_bytes = output_buffer.read()
            output_base64 = base64.b64encode(output_bytes).decode()

            responses.append(VoiceConversionResponse(output_base64=output_base64))
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Error processing sample {idx}: {str(e)}")
    return responses
