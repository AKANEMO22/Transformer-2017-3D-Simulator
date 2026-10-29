# TRANSFORMER 2017 // INTERACTIVE 3D EXECUTION & NODE INSPECTOR

> **Hệ thống Mô Phỏng 3D Tương Tác Kiến Trúc Transformer 2017 Nguyên Bản (*Attention Is All You Need, Vaswani et al.*)**
> Trực tiếp quan sát luồng chuyển hóa tensor, giải mã tự hồi quy và kiểm tra chi tiết từng node tính toán trong không gian 3D.

[![Pytest Tests](https://img.shields.io/badge/pytest-12%20passed-emerald)](https://github.com/AKANEMO22/Transformer-2017-3D-Simulator)
[![Vitest Unit Tests](https://img.shields.io/badge/vitest-11%20passed-cyan)](https://github.com/AKANEMO22/Transformer-2017-3D-Simulator)
[![Architecture Spec](https://img.shields.io/badge/Figure%201-31%20Nodes%20%7C%2041%20Edges-blue)](https://github.com/AKANEMO22/Transformer-2017-3D-Simulator)

---

## 1. Điểm Mới & Tính Năng Trọng Tâm

1. **Nhập liệu tự do & Thực thi mô phỏng linh hoạt (User Input & Simulate)**:
   - Người dùng có thể gõ bất kỳ câu tiếng Việt hoặc tiếng Anh nào vào thanh input và nhấn **`[SIMULATE]`**.
   - Hỗ trợ chọn giữa **`EDUCATIONAL SIMULATION`** (mô phỏng toán học xác định cục bộ) hoặc **`REAL KERAS TRACE`** (forward pass thật từ FastAPI Backend).

2. **Kiểm tra chuyên sâu từng Node (Interactive Node Inspector)**:
   - Nhấp vào bất kỳ khối nào trong **31 nodes kiến trúc** để mở inspector chi tiết:
     - **Inputs & Outputs**: Kích thước tensor `shape`, kiểu dữ liệu `dtype`, và giá trị số thực tế.
     - **Công thức toán học**: Hiển thị phép toán chuẩn mực (ví dụ: $\text{Attention}(Q,K,V)$, LayerNorm, FFN).
     - **Attention Heatmap tương tác**: Chọn Head 0 hoặc Head 1; nhấp vào bất kỳ ô ma trận nào để hiển thị **phép nhân vô hướng thực tế** từng thành phần:
       $$\text{score}[i,j] = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}} \quad \longrightarrow \quad \text{Softmax}(\text{score}) = \text{weight}$$

3. **Điều khiển tiến trình thực thi thời gian thực (Deterministic Timeline Playback)**:
   - Các nút **`PLAY`**, **`PAUSE`**, **`STEP FORWARD`**, **`STEP BACKWARD`**, **`RESET`** và chọn tốc độ $0.5\times / 1\times / 2\times$.
   - Tín hiệu photon chỉ di chuyển trên các cạnh liên quan trực tiếp đến bước tính toán hiện tại.
   - Hỗ trợ nhảy trực tiếp giữa các bước giải mã Decoder (Decode Step 0, 1, 2, 3).

4. **Minh bạch xuất xứ dữ liệu (Transparent Provenance)**:
   - Phân định rõ ràng nhãn nguồn dữ liệu: `REAL KERAS TRACE` vs `EDUCATIONAL SIMULATION`.
   - Nếu Keras backend ngoại tuyến, hệ thống hiển thị thông báo lỗi rõ ràng và cung cấp nút chuyển sang chế độ Educational Simulation thay vì giả mạo dữ liệu.

---

## 2. So Sánh Các Cấu Hình & Nguồn Dữ Liệu

| Đặc tính | Vaswani et al. 2017 (Gốc) | Educational Mini (Demo) | Simulated Trace | Real Keras Trace |
| :--- | :--- | :--- | :--- | :--- |
| **Số tầng (N)** | 6 Encoder / 6 Decoder | 2 Encoder / 2 Decoder | 2 Encoder / 2 Decoder | 2 Encoder / 2 Decoder |
| **Kích thước d_model** | 512 | 8 | 8 | 8 |
| **Số đầu Attention (h)** | 8 (d_k=64) | 2 (d_k=4) | 2 (d_k=4) | 2 (d_k=4) |
| **Kích thước ẩn d_ff** | 2048 | 16 | 16 | 16 |
| **Môi trường chạy** | Phân tán trên 8 GPU P100 | Máy tính cá nhân / Web | Frontend JS/TS (Seed 4242) | Python 3.11 + Keras 3 (TensorFlow) |
| **Mục đích** | Dịch máy WMT 2014 En-De | Giảng dạy trực quan Figure 1 | Trình diễn tức thời, tin cậy cao | Kiểm chứng forward pass tầng Keras thật |

> **Lưu ý khoa học**: Mô hình giáo dục thu nhỏ được khởi tạo trọng số ngẫu nhiên để minh họa hình dạng tensor và phép tính vi kiến trúc. Mô hình chưa qua tiền huấn luyện trên kho ngữ liệu tiếng Việt quy mô lớn nên không phản ánh ngữ nghĩa học máy hoàn chỉnh.

---

## 3. Cấu Trúc Dự Án

```
Transformer_3D_Simulator_Package/
├── transformer_2017_spec.json           # Single Source of Truth: 31 nodes, 41 edges
├── backend/
│   ├── transformer_keras_2017.py        # Mô hình Keras 3 thật (Embedding, MHA, LayerNorm, FFN, Causal, Cross)
│   ├── main.py                          # FastAPI API (/api/health, /api/samples, /api/trace)
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── components/figure1/
│   │   │   ├── Figure1Scene.tsx         # Scene 3D tương tác với OrbitControls
│   │   │   ├── Figure1Node.tsx          # 31 khối kiến trúc với hiệu ứng hover, selected, active
│   │   │   ├── Figure1Edge.tsx          # Dòng truyền tín hiệu photon theo timeline thực thi
│   │   │   ├── NodeInspector.tsx        # Bảng phân tích chi tiết tensor & phép nhân dot-product
│   │   │   ├── Figure1Telemetry.tsx     # Bảng điều khiển Telemetry & danh sách node
│   │   │   ├── Figure1Controls.tsx      # Thanh nhập liệu, SIMULATE, chế độ & playback controls
│   │   │   └── ExplodedLayerDetail.tsx  # Bóc tách chi tiết vi kiến trúc Encoder/Decoder
│   │   ├── services/
│   │   │   ├── api.ts                   # Gọi FastAPI backend & xử lý lỗi minh bạch
│   │   │   ├── demoTrace.ts             # Bộ sinh trace toán học giáo dục có seed xác định
│   │   │   ├── traceValidator.ts        # Kiểm định tính toàn vẹn và chuẩn hóa xác suất
│   │   │   └── executionTimeline.ts     # Bộ điều phối timeline forward pass theo thứ tự phụ thuộc
│   │   ├── types/trace.ts               # Hệ thống kiểu dữ liệu TypeScript
│   │   ├── utils/nodeDetailMapper.ts    # Ánh xạ 31 nodes kiến trúc sang dữ liệu tensor tương ứng
│   │   └── tests/
│   │       └── transformer_frontend.test.ts # Bộ 11 unit test Vitest
│   ├── package.json
│   ├── vite.config.ts
│   └── index.html
├── tests/
│   ├── test_architecture_spec.py        # Kiểm thử cấu trúc đồ thị DAG & tính đồng bộ spec
│   └── test_transformer_3d.py           # Kiểm thử forward pass Keras 3, softmax và causal mask
└── run_3d_simulator.bat                 # Script 1-Click khởi chạy trên Windows
```

---

## 4. Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### Cách 1: Chạy 1-Click (Khuyên dùng trên Windows)
Nhấp đúp chuột vào file `run_3d_simulator.bat`. 
Script sẽ tự động:
1. Khởi động Backend FastAPI tại port 8000.
2. Khởi động Frontend Vite tại port 5173.
3. Mở trình duyệt tại `http://localhost:5173/`.

### Cách 2: Chạy bằng Terminal

**1. Khởi động Backend (Python):**
```powershell
python backend/main.py
```

**2. Khởi động Frontend (React Three Fiber):**
```powershell
cd frontend
npm install
npm run dev
```

Mở trình duyệt: `http://localhost:5173/`

---

## 5. Chạy Kiểm Thử Tự Động (Test Suites)

### Kiểm thử Frontend (Vitest - 11 tests):
```powershell
cd frontend
npm test
```
*Kết quả:*
- Deterministic reproducibility check: **PASSED**
- Input variance verification: **PASSED**
- Attention weight normalization $\sum=1.0$: **PASSED**
- Causal mask future zeroing: **PASSED**
- Cross-attention key dimension alignment: **PASSED**
- 31/31 Architecture nodes inspectability: **PASSED**
- Multi-layer separation (Layer 0 vs Layer 1): **PASSED**
- Execution timeline dependency ordering: **PASSED**
- Decoder step token update: **PASSED**
- Trace provenance & invalid trace rejection: **PASSED**
- Unavailable state fallback: **PASSED**

### Kiểm thử Backend & Kiến trúc (Pytest - 12 tests):
```powershell
python -m pytest tests/ -v
```
*Kết quả:*
- `test_spec_integrity_and_nodes`: **PASSED** (31 nodes)
- `test_edge_count_and_topology`: **PASSED** (41 edges)
- `test_residual_edges_topology`: **PASSED** (Bypass edges)
- `test_memory_edges_cross_attention_topology`: **PASSED** (Encoder $\rightarrow$ Decoder cross-attention)
- `test_frontend_spec_synchronization`: **PASSED**
- `test_keras_model_synchronization_with_spec`: **PASSED**
- `test_positional_encoding_formula`: **PASSED**
- `test_causal_mask_structure`: **PASSED**
- `test_encoder_attention_shapes_and_weights`: **PASSED** (Layer 1 & Layer 2)
- `test_decoder_masked_and_cross_attention`: **PASSED** (Layer 0 & Layer 1)
- `test_linear_logits_and_probabilities`: **PASSED**
- `test_autoregressive_generation`: **PASSED**
