import React, { useState } from 'react';
import type { NodeExecutionTrace, TensorData } from '../../types/trace';
import { Layers, Activity, Cpu, Sparkles, X, ChevronRight, Hash } from 'lucide-react';

interface NodeInspectorProps {
  detail: NodeExecutionTrace;
  onClose?: () => void;
  selectedHead?: number;
  onSelectHead?: (head: number) => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({
  detail,
  onClose,
  selectedHead = 0,
  onSelectHead
}) => {
  const [activeHead, setActiveHead] = useState<number>(selectedHead);
  const [selectedCell, setSelectedCell] = useState<{ queryIdx: number; keyIdx: number } | null>(null);

  const handleHeadChange = (head: number) => {
    setActiveHead(head);
    setSelectedCell(null);
    onSelectHead?.(head);
  };

  const currentHead = detail.attentionHeads?.[activeHead] || detail.attentionHeads?.[0];

  // Compute numerical breakdown for a selected cell
  const getCellCalculation = () => {
    if (!currentHead || !selectedCell) return null;
    const { queryIdx, keyIdx } = selectedCell;
    const qVec = currentHead.q?.[queryIdx] || [];
    const kVec = currentHead.k?.[keyIdx] || [];
    const rawScore = currentHead.rawScores?.[queryIdx]?.[keyIdx];
    const weight = currentHead.weights?.[queryIdx]?.[keyIdx];
    const dK = qVec.length || 4;
    const scale = Math.sqrt(dK);

    let sumDot = 0;
    const terms: string[] = [];
    for (let i = 0; i < Math.min(qVec.length, kVec.length); i++) {
      const prod = qVec[i] * kVec[i];
      sumDot += prod;
      terms.push(`${qVec[i].toFixed(2)} × ${kVec[i].toFixed(2)}`);
    }

    return {
      queryToken: currentHead.queryTokens[queryIdx] || `q_${queryIdx}`,
      keyToken: currentHead.keyTokens[keyIdx] || `k_${keyIdx}`,
      qVec,
      kVec,
      terms,
      sumDot: parseFloat(sumDot.toFixed(4)),
      dK,
      scale: parseFloat(scale.toFixed(4)),
      scaledScore: typeof rawScore === 'number' ? rawScore : parseFloat((sumDot / scale).toFixed(4)),
      weight: typeof weight === 'number' ? weight : 0
    };
  };

  const cellCalc = getCellCalculation();

  if (detail.status === 'unavailable') {
    return (
      <div className="bg-slate-900/95 border border-red-900/50 rounded-xl p-4 text-slate-300 backdrop-blur-md shadow-2xl">
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center gap-2 text-red-400 font-mono text-sm font-semibold">
            <Activity className="w-4 h-4 text-red-400" />
            <span>TRACE UNAVAILABLE</span>
          </div>
          {onClose && (
            <button onClick={onClose} className="p-1 hover:bg-slate-800 rounded text-slate-400">
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        <div className="text-xs text-slate-400 leading-relaxed font-mono">
          {detail.unavailableReason || 'Không có dữ liệu tensor tại node này.'}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-950/95 border border-cyan-500/40 rounded-xl p-4 text-slate-200 backdrop-blur-xl shadow-2xl max-w-full text-xs font-mono select-text max-h-[85vh] overflow-y-auto">
      {/* HEADER */}
      <div className="flex items-start justify-between border-b border-slate-800 pb-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-cyan-950 text-cyan-400 border border-cyan-800/80">
              {detail.nodeId}
            </span>
            {detail.layerIndex !== undefined && (
              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-indigo-950 text-indigo-400 border border-indigo-800/80">
                Layer {detail.layerIndex}
              </span>
            )}
            {detail.decoderStep !== undefined && (
              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-400 border border-amber-800/80">
                Decode Step {detail.decoderStep}
              </span>
            )}
          </div>
          <h3 className="text-sm font-bold text-white mt-1 flex items-center gap-1.5">
            <Cpu className="w-4 h-4 text-cyan-400 inline" />
            {detail.nodeName || detail.nodeId}
          </h3>
          <p className="text-[11px] text-slate-400 mt-0.5">{detail.operation}</p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors"
            title="Đóng panel"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* FORMULA & DESCRIPTION */}
      {detail.formula && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2.5 mb-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1 font-semibold">Công thức toán học</div>
          <div className="text-cyan-300 font-semibold text-xs tracking-wide overflow-x-auto py-0.5">
            {detail.formula}
          </div>
          {detail.description && (
            <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-sans">{detail.description}</p>
          )}
        </div>
      )}

      {/* SPECIALIZED ATTENTION HEATMAP WITH HEAD SELECTION & CELL CLICK BREAKDOWN */}
      {detail.attentionHeads && detail.attentionHeads.length > 0 && (
        <div className="bg-slate-900/90 border border-cyan-900/60 rounded-lg p-3 mb-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-cyan-400 uppercase tracking-wider font-semibold flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-cyan-400" />
              Attention Matrix & Head Inspection
            </span>
            {/* Head Switcher */}
            <div className="flex gap-1">
              {detail.attentionHeads.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => handleHeadChange(idx)}
                  className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all ${
                    activeHead === idx
                      ? 'bg-cyan-500 text-black shadow-lg shadow-cyan-500/20'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }`}
                >
                  Head {idx}
                </button>
              ))}
            </div>
          </div>

          {currentHead && (
            <div>
              <div className="text-[10px] text-slate-400 mb-1.5">
                Nhấp vào từng ô để xem chi tiết phép nhân vô hướng Q·K^T / √d_k:
              </div>
              <div className="overflow-x-auto pb-1">
                <table className="border-collapse text-[10px]">
                  <thead>
                    <tr>
                      <th className="p-1 text-slate-500 text-right pr-2">Q \ K</th>
                      {currentHead.keyTokens.map((kTok, kIdx) => (
                        <th key={kIdx} className="p-1 text-center font-mono text-cyan-300/90 min-w-10">
                          {kTok}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {currentHead.weights.map((row, qIdx) => (
                      <tr key={qIdx}>
                        <td className="p-1 text-right font-mono text-amber-300/90 pr-2 whitespace-nowrap">
                          {currentHead.queryTokens[qIdx]}
                        </td>
                        {row.map((weight, kIdx) => {
                          const isSelected = selectedCell?.queryIdx === qIdx && selectedCell?.keyIdx === kIdx;
                          const intensity = Math.min(1, Math.max(0, weight));
                          const bgStyle = {
                            backgroundColor: `rgba(6, 182, 212, ${Math.max(0.08, intensity * 0.85)})`
                          };
                          return (
                            <td key={kIdx} className="p-0.5">
                              <button
                                onClick={() => setSelectedCell({ queryIdx: qIdx, keyIdx: kIdx })}
                                style={bgStyle}
                                className={`w-full py-1 px-1.5 rounded text-center transition-all cursor-pointer font-bold ${
                                  isSelected
                                    ? 'ring-2 ring-white text-white font-extrabold scale-105 z-10 shadow-lg shadow-cyan-500/50'
                                    : intensity > 0.4
                                    ? 'text-white'
                                    : 'text-slate-300 hover:ring-1 hover:ring-cyan-400'
                                }`}
                                title={`Query: ${currentHead.queryTokens[qIdx]} -> Key: ${currentHead.keyTokens[kIdx]} | Weight: ${weight.toFixed(4)}`}
                              >
                                {weight.toFixed(2)}
                              </button>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* CELL CALCULATION BREAKDOWN */}
              {cellCalc && (
                <div className="mt-3 p-2.5 bg-slate-950 border border-cyan-500/40 rounded-lg text-[11px] leading-relaxed">
                  <div className="text-[10px] text-amber-400 font-bold uppercase tracking-wider mb-1 flex items-center gap-1">
                    <Hash className="w-3 h-3 text-amber-400" />
                    Tính toán chi tiết ô [{cellCalc.queryToken} → {cellCalc.keyToken}]:
                  </div>
                  <div className="text-slate-300 space-y-1 font-mono text-[10px]">
                    <div>
                      <span className="text-slate-500">Vector Q [{cellCalc.queryToken}]:</span>{' '}
                      [{cellCalc.qVec.map(v => v.toFixed(2)).join(', ')}]
                    </div>
                    <div>
                      <span className="text-slate-500">Vector K [{cellCalc.keyToken}]:</span>{' '}
                      [{cellCalc.kVec.map(v => v.toFixed(2)).join(', ')}]
                    </div>
                    <div className="pt-1 text-cyan-300">
                      <span className="text-slate-400">Tích vô hướng:</span> Q·K^T = {cellCalc.terms.join(' + ')} ={' '}
                      <span className="font-bold text-white">{cellCalc.sumDot}</span>
                    </div>
                    <div className="text-cyan-300">
                      <span className="text-slate-400">Chia tỷ lệ:</span> score = {cellCalc.sumDot} / √{cellCalc.dK} ={' '}
                      <span className="font-bold text-white">{cellCalc.scaledScore.toFixed(3)}</span>
                    </div>
                    <div className="text-emerald-400 font-bold">
                      <span className="text-slate-400">Softmax row normalized:</span> weight ={' '}
                      <span className="text-white text-xs">{cellCalc.weight.toFixed(4)}</span> (
                      {(cellCalc.weight * 100).toFixed(1)}%)
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* INPUT TENSORS */}
      {detail.inputs.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 font-semibold flex items-center gap-1">
            <ChevronRight className="w-3 h-3 text-cyan-400" />
            Input Tensors
          </div>
          <div className="space-y-1.5">
            {detail.inputs.map((inp, idx) => (
              <TensorBlock key={idx} tensor={inp} />
            ))}
          </div>
        </div>
      )}

      {/* INTERMEDIATE TENSORS */}
      {detail.intermediate && detail.intermediate.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 font-semibold flex items-center gap-1">
            <ChevronRight className="w-3 h-3 text-indigo-400" />
            Intermediate Computations
          </div>
          <div className="space-y-1.5">
            {detail.intermediate.map((t, idx) => (
              <TensorBlock key={idx} tensor={t} />
            ))}
          </div>
        </div>
      )}

      {/* OUTPUT TENSORS */}
      {detail.outputs.length > 0 && (
        <div className="mb-2">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 font-semibold flex items-center gap-1">
            <ChevronRight className="w-3 h-3 text-emerald-400" />
            Output Tensors
          </div>
          <div className="space-y-1.5">
            {detail.outputs.map((out, idx) => (
              <TensorBlock key={idx} tensor={out} isOutput />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

interface TensorBlockProps {
  tensor: TensorData;
  isOutput?: boolean;
}

const TensorBlock: React.FC<TensorBlockProps> = ({ tensor, isOutput }) => {
  const [expanded, setExpanded] = useState(false);
  const shapeStr = `[${tensor.shape.join(', ')}]`;

  // Render preview or value
  const renderPreview = () => {
    if (!tensor.values || (Array.isArray(tensor.values) && tensor.values.length === 0)) {
      return <span className="text-slate-500">trống</span>;
    }

    if (tensor.dtype === 'string' || tensor.dtype === 'string[]') {
      return (
        <div className="flex flex-wrap gap-1">
          {Array.isArray(tensor.values) ? (
            tensor.values.map((v, i) => (
              <span key={i} className="px-1.5 py-0.5 rounded bg-cyan-950/70 border border-cyan-800 text-cyan-300 text-[10px]">
                {String(v)}
              </span>
            ))
          ) : (
            <span className="text-white font-sans">{String(tensor.values)}</span>
          )}
        </div>
      );
    }

    // 1D array of numbers
    if (Array.isArray(tensor.values) && typeof tensor.values[0] === 'number') {
      const nums = tensor.values as number[];
      return (
        <div className="text-slate-300 font-mono text-[10px] leading-relaxed">
          [{nums.slice(0, 8).map(n => n.toFixed(2)).join(', ')}
          {nums.length > 8 ? ` ... (${nums.length} phần tử)` : ''}]
        </div>
      );
    }

    // 2D matrix
    if (Array.isArray(tensor.values) && Array.isArray(tensor.values[0])) {
      const mat = tensor.values as number[][];
      const rows = mat.slice(0, expanded ? mat.length : 3);
      return (
        <div className="space-y-0.5 font-mono text-[9px] text-slate-300 bg-slate-950/80 p-1.5 rounded border border-slate-800/80 overflow-x-auto">
          {rows.map((row, rIdx) => (
            <div key={rIdx} className="whitespace-nowrap">
              row {rIdx}: [{row.slice(0, 8).map(v => typeof v === 'number' ? v.toFixed(2) : String(v)).join(', ')}
              {row.length > 8 ? ' ...' : ''}]
            </div>
          ))}
          {mat.length > 3 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-[9px] text-cyan-400 hover:underline pt-0.5 cursor-pointer block"
            >
              {expanded ? '▲ Thu gọn' : `▼ Xem thêm ${mat.length - 3} hàng`}
            </button>
          )}
        </div>
      );
    }

    return (
      <pre className="text-[9px] text-slate-400 overflow-x-auto max-h-24">
        {JSON.stringify(tensor.values, null, 1)}
      </pre>
    );
  };

  return (
    <div className={`p-2 rounded-lg border text-xs ${
      isOutput
        ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-200'
        : 'bg-slate-900/60 border-slate-800 text-slate-300'
    }`}>
      <div className="flex items-center justify-between mb-1">
        <span className="font-bold text-white flex items-center gap-1.5">
          <Layers className="w-3 h-3 text-slate-400" />
          {tensor.name}
        </span>
        <span className="text-[10px] text-slate-500 font-mono">
          {tensor.dtype} {shapeStr}
        </span>
      </div>
      {tensor.description && (
        <div className="text-[10px] text-slate-400 mb-1 leading-tight font-sans">
          {tensor.description}
        </div>
      )}
      <div className="mt-1">{renderPreview()}</div>
    </div>
  );
};
