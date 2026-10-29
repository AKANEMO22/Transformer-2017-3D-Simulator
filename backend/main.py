"""
FastAPI Server providing real Keras 3 Transformer 2017 execution traces to the 3D frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import os

from backend.transformer_keras_2017 import EducationalTransformer2017

app = FastAPI(
    title="Transformer 2017 3D Simulator API",
    description="Backend powered by Keras 3 & TensorFlow extracting genuine forward-pass traces.",
    version="2.0.0"
)

# CORS: Local development allowlist without unsafe wildcard credentials
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize educational model
print("Initializing Keras 3 Transformer 2017 Model...")
model = EducationalTransformer2017(d_model=8, num_heads=2, d_ff=16)
print("Model initialized successfully.")


class TraceRequest(BaseModel):
    source_text: str = Field(..., min_length=1, max_length=120)
    target_prompt: Optional[str] = None
    max_decode_steps: Optional[int] = Field(default=4, ge=1, le=8)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "backend": "Keras 3 + TensorFlow",
        "model_name": "EducationalTransformer2017",
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "d_k": model.d_model // model.num_heads,
        "d_ff": model.d_ff,
        "vocab_size": model.vocab_size,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "notes": "Untrained educational model for architectural demonstration"
    }


@app.get("/api/samples")
def get_samples():
    return {
        "samples": [
            {
                "id": "interview_good",
                "label": "AI Interview - Giải quyết mâu thuẫn (Mẫu)",
                "text": "Lắng nghe ý kiến nhóm giải quyết mâu thuẫn",
                "description": "Câu trả lời đúng trọng tâm thể hiện kỹ năng lắng nghe và hòa giải nhóm."
            },
            {
                "id": "interview_contrast",
                "label": "AI Interview - Lạc đề kỹ thuật (Đối chứng)",
                "text": "Tôi biết python sql mô hình code",
                "description": "Câu trả lời lạc đề, chỉ liệt kê công nghệ mà không trả lời về xử lý mâu thuẫn."
            },
            {
                "id": "classic_2017",
                "label": "Transformer 2017 - Attention Is All You Need",
                "text": "Ý kiến thảo luận giải quyết đồng thuận",
                "description": "Kịch bản tiêu chuẩn kiểm chứng luồng Encoder - Masked Decoder - Cross Attention."
            }
        ]
    }


@app.post("/api/trace")
def compute_trace(req: TraceRequest):
    if not req.source_text or not req.source_text.strip():
        raise HTTPException(status_code=400, detail="source_text cannot be empty")

    try:
        trace_data = model.run_trace(
            source_text=req.source_text.strip(),
            target_prompt=req.target_prompt,
            max_decode_steps=req.max_decode_steps or 4
        )
        return trace_data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Sanitize exception message to avoid exposing internal directory paths
        err_msg = str(e).split("\n")[0]
        raise HTTPException(status_code=500, detail=f"Lỗi tính toán mô hình Keras 3: {err_msg}")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
