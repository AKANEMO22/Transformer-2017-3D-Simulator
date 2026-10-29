import type { TransformerTrace, TraceMetadata } from '../types/trace';
import { generateDemoTrace } from './demoTrace';
import { validateTransformerTrace } from './traceValidator';

const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8000';

export class ApiError extends Error {
  status?: number;
  canFallbackToSim: boolean;

  constructor(message: string, status?: number, canFallbackToSim = true) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.canFallbackToSim = canFallbackToSim;
  }
}

/**
 * Fetch a genuine Keras 3 forward-pass trace from the FastAPI backend.
 * Never silently returns fake data or fallbacks if the backend is down.
 */
export async function fetchKerasTrace(
  sourceText: string,
  targetPrompt?: string,
  maxDecodeSteps = 4
): Promise<TransformerTrace> {
  const url = `${API_BASE}/api/trace`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        source_text: sourceText,
        target_prompt: targetPrompt,
        max_decode_steps: maxDecodeSteps
      })
    });
  } catch (netErr: any) {
    throw new ApiError(
      `Không kết nối được Keras backend tại ${API_BASE}. Bạn có thể chuyển sang chế độ Educational Simulation để tiếp tục.`,
      0,
      true
    );
  }

  if (!response.ok) {
    let errorDetail = `Lỗi máy chủ (${response.status})`;
    try {
      const errJson = await response.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // ignore json parse error
    }
    throw new ApiError(errorDetail, response.status, true);
  }

  const rawData: TransformerTrace = await response.json();

  // Attach or verify metadata
  if (!rawData.metadata) {
    const meta: TraceMetadata = {
      runId: `keras_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      source: 'keras',
      inputText: sourceText,
      modelName: 'Keras 3 Educational Transformer (Vaswani 2017)',
      modelConfig: {
        numEncoderLayers: 2,
        numDecoderLayers: 2,
        dModel: rawData.d_model || 8,
        numHeads: rawData.num_heads || 2,
        dFF: rawData.d_ff || 16,
        vocabSize: rawData.vocab?.length || 25
      },
      createdAt: new Date().toISOString(),
      isUntrained: true,
      disclaimer: 'Trace được trích xuất từ các layer Keras 3 thật (Embedding, MHA, LayerNorm, FFN, Causal Mask, Cross-Attention).'
    };
    rawData.metadata = meta;
  } else {
    rawData.metadata.source = 'keras';
  }

  // Validate trace
  const validation = validateTransformerTrace(rawData);
  if (!validation.valid) {
    throw new ApiError(`Dữ liệu Keras trace không hợp lệ: ${validation.errors.join(', ')}`, 502, true);
  }

  return rawData;
}

/**
 * Fetch sample demonstration sentences from the backend or fallback presets.
 */
export async function fetchSamples(): Promise<Array<{ id: string; label: string; text: string; description: string }>> {
  try {
    const res = await fetch(`${API_BASE}/api/samples`);
    if (res.ok) {
      const data = await res.json();
      if (data.samples && Array.isArray(data.samples)) {
        return data.samples;
      }
    }
  } catch {
    // Backend offline; return standard samples
  }

  return [
    {
      id: "interview_good",
      label: "AI Interview - Giải quyết mâu thuẫn (Mẫu)",
      text: "Lắng nghe ý kiến nhóm giải quyết mâu thuẫn",
      description: "Câu trả lời đúng trọng tâm thể hiện kỹ năng lắng nghe và hòa giải nhóm."
    },
    {
      id: "interview_contrast",
      label: "AI Interview - Lạc đề kỹ thuật (Đối chứng)",
      text: "Tôi biết python sql mô hình code",
      description: "Câu trả lời lạc đề, chỉ liệt kê công nghệ mà không trả lời về xử lý mâu thuẫn."
    },
    {
      id: "classic_2017",
      label: "Transformer 2017 - Attention Is All You Need",
      text: "Ý kiến thảo luận giải quyết đồng thuận",
      description: "Kịch bản tiêu chuẩn kiểm chứng luồng Encoder - Masked Decoder - Cross Attention."
    }
  ];
}

/**
 * Check backend health status without throwing an unhandled exception.
 */
export async function checkBackendHealth(): Promise<{ isOnline: boolean; info?: any }> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
    if (res.ok) {
      const info = await res.json();
      return { isOnline: true, info };
    }
  } catch {
    // Offline
  }
  return { isOnline: false };
}

/**
 * Execute trace generation based on explicit simulation mode selected by user.
 */
export async function executeTrace(
  sourceText: string,
  mode: 'keras' | 'simulated'
): Promise<TransformerTrace> {
  if (mode === 'keras') {
    return await fetchKerasTrace(sourceText);
  } else {
    const trace = generateDemoTrace(sourceText);
    const validation = validateTransformerTrace(trace);
    if (!validation.valid) {
      throw new ApiError(`Mô phỏng toán học không hợp lệ: ${validation.errors.join(', ')}`, 500, false);
    }
    return trace;
  }
}
