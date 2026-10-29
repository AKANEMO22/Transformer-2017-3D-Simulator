# TRANSFORMER 2017 // 3D VISUAL SIMULATOR (FIGURE 1 & ARGON-TX7)

Hệ thống mô phỏng 3D tương tác kiến trúc Transformer 2017 nguyên bản (*Attention Is All You Need, Vaswani et al.*) được xây dựng trên nền tảng **Keras 3 Backend** và **React Three Fiber (R3F) Frontend**.

Mô phỏng tuân thủ 100% sơ đồ Figure 1 của bài báo gốc với cấu trúc đặc tả tiêu chuẩn khoa học (`transformer_2017_spec.json`), hỗ trợ cả chế độ chuẩn mực hình học **Figure 1 3D Architecture Simulator** và chế độ lắp ráp phần cứng công nghiệp **ARGON-TX7 Module Assembly**.

---

## 1. Cấu Trúc Dự Án

```
Transformer_3D_Simulator_Package/
│
├── transformer_2017_spec.json            # Đặc tả kiến trúc máy đọc (Single Source of Architectural Truth)
│                                         # 26 nodes, 38 edges (10 residual, 2 cross-attn memory, 2 causal mask)
│
├── backend/                              # Backend tính toán Keras 3 & FastAPI
│   ├── transformer_keras_2017.py         # Transformer 2017 nguyên bản (Q, K, V, Softmax, Causal, Cross-Attn)
│   ├── main.py                           # Server FastAPI cung cấp API /api/trace & /api/architecture
│   └── __init__.py
│
├── frontend/                             # Frontend React Three Fiber + Vite + Tailwind
│   ├── src/
│   │   ├── spec/
│   │   │   └── transformer_2017_spec.json# Bản sao spec kiến trúc phục vụ render 3D
│   │   ├── components/
│   │   │   ├── figure1/                  # Module 3D Figure 1 Architecture Simulator
│   │   │   │   ├── Figure1Scene.tsx      # Scene 3D không gian biểu diễn Figure 1
│   │   │   │   ├── Figure1Node.tsx       # Khối 3D biểu diễn node (Embedding, MHA, Add&Norm, FFN, Softmax)
│   │   │   │   ├── Figure1Edge.tsx       # Đường truyền tensor, residual bypass & memory routing
│   │   │   │   ├── ExplodedLayerDetail.tsx # Chi tiết vi kiến trúc bóc tách N=6 sublayers
│   │   │   │   ├── Figure1Telemetry.tsx  # Bảng đo đạc tensor Keras 3 thật & ma trận Attention
│   │   │   │   └── Figure1Controls.tsx   # Thanh điều hướng góc nhìn & chế độ mô phỏng
│   │   │   └── argon/                    # Module 3D ARGON-TX7 Hardware Assembly
│   │   │       ├── BasePlatform.tsx      # Khung gầm Graphite với rãnh bus quang học chìm
│   │   │       ├── EmbeddingModule.tsx   # Module ma trận Embedding [5x8-D] với 5 chip token
│   │   │       ├── PositionalModule.tsx  # Bộ cộng hưởng sóng điều hòa sin/cos thạch anh
│   │   │       ├── EncoderBlockModule.tsx# Khối Encoder (Multi-Head Attention, Add&Norm, FFN)
│   │   │       ├── SignalBus.tsx         # Hạt photon phát sáng chạy dọc cáp bus
│   │   │       ├── AssemblyScene.tsx     # Scene 3D chính với góc nhìn Isometric
│   │   │       ├── ExecutionFeed.tsx     # Cột trái 25% (Bus telemetry & code Keras 3)
│   │   │       └── StatusBays.tsx        # Thanh đáy 4 module bays & timeline scrubber
│   │   ├── services/api.ts               # Kết nối API Backend & Fallback tính toán chuẩn
│   │   ├── types/trace.ts                # Định nghĩa kiểu dữ liệu TypeScript
│   │   ├── App.tsx                       # Bộ chuyển đổi linh hoạt giữa Figure 1 và Argon
│   │   └── index.css                     # Dark theme & styling
│   ├── package.json
│   ├── vite.config.ts
│   └── index.html
│
├── tests/                                # Test suite kiểm thử toán học & cấu trúc (9/9 PASSED)
│   ├── test_architecture_spec.py         # Kiểm chứng đồ thị DAG, residual bypass, cross-attn memory
│   └── test_transformer_3d.py            # Kiểm chứng forward pass Keras 3, Attention weights, Causal mask
│
├── screenshots/                          # Bằng chứng nghiệm thu từ Playwright E2E
│   ├── figure1_full_architecture.png     # Toàn cảnh sơ đồ Figure 1 3D
│   ├── figure1_encoder_exploded.png      # Chế độ bóc tách Encoder Layer (Exploded View)
│   ├── figure1_decoder_exploded.png      # Chế độ bóc tách Decoder Layer (Exploded View)
│   ├── figure1_cross_attention.png       # Dòng tín hiệu Memory Cross-Attention từ Encoder sang Decoder
│   ├── figure1_causal_mask.png           # Cơ chế Causal Mask [-inf] ngăn rò rỉ token tương lai
│   ├── figure1_execution_flow.png        # Luồng dữ liệu chạy từng bước forward pass
│   ├── argon_transformer_milestone1.png  # Toàn cảnh giao diện ARGON-TX7 3D Assembly
│   └── argon_transformer_codeview.png    # Cột trái 25% xem mã nguồn Keras 3 thật
│
├── run_3d_simulator.bat                  # Script 1-Click tự động chạy cả Backend & Frontend
└── README.md                             # Tài liệu hướng dẫn chi tiết
```

---

## 2. Hướng Dẫn Khởi Chạy Nhanh

### Cách 1: Chạy 1-Click (Khuyên dùng trên Windows)
Nhấp đúp chuột vào file `run_3d_simulator.bat`. 
Script sẽ tự động:
1. Bật Backend FastAPI tại port 8000.
2. Cài đặt `npm install` (nếu chưa có) và khởi động Frontend Vite tại port 5173.
3. Mở sẵn trình duyệt tại `http://localhost:5173/`.

### Cách 2: Chạy Thủ Công Bằng Terminal

**Terminal 1 (Backend Keras 3):**
```powershell
python backend/main.py
```

**Terminal 2 (Frontend React Three Fiber):**
```powershell
cd frontend
npm install
npm run dev
```

Truy cập: `http://localhost:5173/`

---

## 3. Chạy Kiểm Thử Toán Học & Cấu Trúc (Pytest)

Dự án có bộ test tự động kiểm chứng 100% tính khoa học và cấu trúc đồ thị:

```powershell
pytest tests/ -v
```

**Kết quả kiểm thử:**
- `tests/test_architecture_spec.py`:
  - `test_node_counts_and_types`: PASSED (Đúng 26 node, đầy đủ Encoder, Decoder, Output)
  - `test_sublayer_residual_connections`: PASSED (100% sublayer có đường tắt nối vào Add & Norm)
  - `test_cross_attention_memory_edges`: PASSED (Keys & Values nhận từ Encoder; Queries từ Decoder)
  - `test_causal_mask_applied_to_masked_mha`: PASSED (Causal mask được nối chính xác)
- `tests/test_transformer_3d.py`:
  - `test_positional_encoding_formula`: PASSED (Công thức sin/cos chuẩn Vaswani et al.)
  - `test_causal_mask_structure`: PASSED (Ma trận tam giác trên gán -1e9)
  - `test_encoder_attention_shapes_and_weights`: PASSED (Softmax sum = 1.0, shape chuẩn)
  - `test_decoder_masked_and_cross_attention`: PASSED (Decoder self & cross attention)
  - `test_autoregressive_generation`: PASSED (Sinh token tự hồi quy từng bước)

---

## 4. Các Chế Độ Tương Tác Trong Simulator

1. **Figure 1 3D Architecture Simulator (Mặc định)**:
   - **`[FIGURE 1 SCHEMATIC]`**: Chế độ phối cảnh 3D không gian mở theo đúng nguyên mẫu sơ đồ Figure 1 (bài báo Vaswani et al. 2017).
   - **`[ENCODER EXPLODED]`**: Bóc tách chi tiết khối Encoder với Multi-Head Attention, Add & Norm và Feed-Forward Network.
   - **`[DECODER EXPLODED]`**: Bóc tách chi tiết khối Decoder với Masked Multi-Head Attention, Cross-Attention và Feed-Forward Network.
   - **`[STEP FORWARD / BACKWARD]`**: Bước qua từng chặng trong forward pass, quan sát luồng tín hiệu di chuyển từ Input đến Output.
   - **`[INSPECT TENSOR]`**: Bảng telemetry hiển thị hình dạng Tensor, kiểu dữ liệu, các giá trị xác thực Keras 3 runtime và ma trận Attention Head heatmap.

2. **ARGON-TX7 Hardware Assembly**:
   - Nhấn nút chuyển đổi chế độ ở góc trên để trải nghiệm giao diện lắp ráp module vi mạch phần cứng công nghiệp ARGON A7.
