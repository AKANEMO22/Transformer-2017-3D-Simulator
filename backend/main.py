"""
FastAPI Server providing real Keras 3 Transformer 2017 execution traces to the 3D frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from transformer_keras_2017 import EducationalTransformer2017

app = FastAPI(
    title="Transformer 2017 3D Simulator API",
    description="Backend powered by Keras 3 & TensorFlow extracting genuine forward-pass traces.",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize educational model
print("Initializing Keras 3 Transformer 2017 Model...")
model = EducationalTransformer2017(d_model=8, num_heads=2, d_ff=16)
print("Model initialized successfully.")


class TraceRequest(BaseModel):
    source_text: str
    target_prompt: Optional[str] = None
    max_decode_steps: Optional[int] = 4


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "backend": "Keras 3 + TensorFlow",
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "vocab_size": model.vocab_size
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
    if not req.source_text.strip():
        raise HTTPException(status_code=400, detail="source_text cannot be empty")
    try:
        trace_data = model.run_trace(
            source_text=req.source_text,
            target_prompt=req.target_prompt,
            max_decode_steps=req.max_decode_steps or 4
        )
        return trace_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
