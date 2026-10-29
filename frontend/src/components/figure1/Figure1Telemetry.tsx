import React, { useState } from 'react';
import { Activity, BookOpen, ChevronLeft, ChevronRight, Layers } from 'lucide-react';
import type { TransformerTrace } from '../../types/trace';
import { getNodeDetail } from '../../utils/nodeDetailMapper';
import { NodeInspector } from './NodeInspector';
import specData from '../../spec/transformer_2017_spec.json';

interface Figure1TelemetryProps {
  trace: TransformerTrace | null;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  isExecuting?: boolean;
  activeDecoderStep?: number;
  currentExecutionStep?: number;
  totalExecutionSteps?: number;
}

export const Figure1Telemetry: React.FC<Figure1TelemetryProps> = ({
  trace,
  selectedNodeId,
  onSelectNode,
  isExecuting = false,
  activeDecoderStep = 0,
  currentExecutionStep = 0,
  totalExecutionSteps = 18
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedHead, setSelectedHead] = useState(0);

  const selectedNodeDetail = selectedNodeId
    ? getNodeDetail(selectedNodeId, trace, activeDecoderStep)
    : null;

  const metadata = trace?.metadata;
  const isKeras = metadata?.source === 'keras';
  const dModel = trace?.d_model || 8;
  const numHeads = trace?.num_heads || 2;
  const dK = trace?.d_k || 4;
  const dFF = trace?.d_ff || 16;
  const encTokens = trace?.encoder?.source_tokens || [];

  if (collapsed) {
    return (
      <div className="absolute top-4 left-4 z-20">
        <button
          onClick={() => setCollapsed(false)}
          className="flex items-center gap-1.5 px-3 py-2 bg-slate-900/90 border border-slate-700/80 rounded-xl text-xs font-mono text-cyan-400 hover:text-white hover:bg-slate-800 transition-all shadow-xl backdrop-blur-md"
          title="Mở thanh Telemetry & Inspector"
        >
          <ChevronRight className="w-4 h-4" />
          <span>INSPECTOR</span>
          {selectedNodeId && (
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          )}
        </button>
      </div>
    );
  }

  return (
    <div className="w-[340px] lg:w-[380px] h-full flex flex-col bg-[#080d1a]/95 border-r border-slate-800/80 shadow-2xl z-10 select-none backdrop-blur-md">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between bg-[#0b1222]">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold tracking-wider text-slate-100 uppercase">
            {selectedNodeId ? 'NODE INSPECTOR' : 'SIMULATION TELEMETRY'}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {selectedNodeId && (
            <button
              onClick={() => onSelectNode(null)}
              className="text-[10px] text-slate-400 hover:text-white px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono transition-colors"
            >
              Overview
            </button>
          )}
          <button
            onClick={() => setCollapsed(true)}
            className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition-colors"
            title="Thu gọn bảng điều khiển"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-3 text-xs font-mono space-y-3">
        {selectedNodeDetail ? (
          <NodeInspector
            detail={selectedNodeDetail}
            onClose={() => onSelectNode(null)}
            selectedHead={selectedHead}
            onSelectHead={setSelectedHead}
          />
        ) : (
          /* OVERVIEW MODE (WHEN NO NODE SELECTED) */
          <div className="space-y-3">
            {/* Provenance Badge */}
            <div className={`p-3 rounded-xl border ${
              isKeras
                ? 'bg-emerald-950/20 border-emerald-500/40'
                : 'bg-cyan-950/20 border-cyan-500/40'
            }`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className={`text-[11px] font-bold tracking-wide flex items-center gap-1.5 ${
                  isKeras ? 'text-emerald-400' : 'text-cyan-400'
                }`}>
                  <Activity className="w-3.5 h-3.5" />
                  {isKeras ? 'REAL KERAS 3 TRACE' : 'EDUCATIONAL SIMULATION'}
                </span>
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${
                  isExecuting ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 animate-pulse' : 'bg-slate-800 text-slate-400'
                }`}>
                  {isExecuting ? 'RUNNING' : 'IDLE'}
                </span>
              </div>
              <p className="text-[10px] text-slate-300 font-sans leading-relaxed">
                {metadata?.disclaimer ||
                  (isKeras
                    ? 'Dữ liệu được trích xuất trực tiếp từ forward pass thực tế của mô hình Keras 3 trong backend.'
                    : 'Dữ liệu mô phỏng toán học chuẩn mực theo Figure 1 bài báo Vaswani et al. 2017.')}
              </p>
            </div>

            {/* Model Dimensions */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Cấu hình tham số kích thước
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="p-1.5 rounded bg-black/40 border border-slate-800">
                  <div className="text-slate-400 text-[10px]">d_model</div>
                  <div className="text-white font-bold">{dModel}</div>
                </div>
                <div className="p-1.5 rounded bg-black/40 border border-slate-800">
                  <div className="text-slate-400 text-[10px]">Attention Heads</div>
                  <div className="text-cyan-400 font-bold">{numHeads} (d_k={dK})</div>
                </div>
                <div className="p-1.5 rounded bg-black/40 border border-slate-800">
                  <div className="text-slate-400 text-[10px]">d_ff</div>
                  <div className="text-white font-bold">{dFF}</div>
                </div>
                <div className="p-1.5 rounded bg-black/40 border border-slate-800">
                  <div className="text-slate-400 text-[10px]">Layers (N)</div>
                  <div className="text-amber-400 font-bold">2 Enc / 2 Dec</div>
                </div>
              </div>
            </div>

            {/* Current Input & Tokens */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Chuỗi đầu vào (Source)
              </div>
              <div className="text-slate-200 text-xs font-sans font-medium">
                "{trace?.source_text || 'Chưa nhập câu'}"
              </div>
              {encTokens.length > 0 && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {encTokens.map((tok, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-800/80 text-cyan-300 text-[10px]"
                    >
                      {tok}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Execution Progress */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                <span>Tiến trình Forward Pass</span>
                <span className="text-cyan-400 font-bold">
                  {currentExecutionStep + 1} / {totalExecutionSteps}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-cyan-500 to-indigo-500 h-full transition-all duration-300"
                  style={{
                    width: `${((currentExecutionStep + 1) / Math.max(1, totalExecutionSteps)) * 100}%`
                  }}
                />
              </div>
              <div className="text-[10px] text-slate-400">
                Nhấp vào bất kỳ khối 3D nào để xem chi tiết tensor, công thức và ma trận chú ý.
              </div>
            </div>

            {/* Quick Node Navigator */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold flex items-center gap-1">
                <Layers className="w-3 h-3 text-slate-400" />
                <span>31 Khối Kiến Trúc Figure 1</span>
              </div>
              <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
                {specData.nodes.map(n => (
                  <button
                    key={n.id}
                    onClick={() => onSelectNode(n.id)}
                    className="w-full text-left p-1.5 rounded bg-black/40 hover:bg-slate-800/90 border border-slate-800/80 flex items-center justify-between text-[10px] transition-colors group"
                  >
                    <span className="text-slate-300 group-hover:text-cyan-300 truncate">
                      {n.label}
                    </span>
                    <span className="text-[9px] text-slate-500 font-mono">
                      {n.column}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
