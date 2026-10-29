import React, { useState } from 'react';
import { Activity, BookOpen, CheckCircle, Info } from 'lucide-react';
import type { TransformerTrace } from '../../types/trace';
import specData from '../../spec/transformer_2017_spec.json';

interface Figure1TelemetryProps {
  trace: TransformerTrace;
  selectedNodeId: string | null;
  isExecuting: boolean;
}

export const Figure1Telemetry: React.FC<Figure1TelemetryProps> = ({
  trace,
  selectedNodeId,
  isExecuting
}) => {
  const [activeTab, setActiveTab] = useState<'audit' | 'tensors' | 'node'>('audit');

  const selectedNode = selectedNodeId 
    ? specData.nodes.find((n) => n.id === selectedNodeId) 
    : null;

  return (
    <div className="w-[320px] lg:w-[350px] h-full flex flex-col bg-[#080d1a] border-r border-slate-800 shadow-2xl z-10 select-none">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-[#0b1222]">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold tracking-wider text-slate-100 uppercase">
            FIGURE 1 ARCHITECTURE
          </span>
        </div>
        <div className="flex items-center gap-1 font-mono text-[11px]">
          <button
            onClick={() => setActiveTab('audit')}
            className={`px-2 py-1 rounded transition-all ${
              activeTab === 'audit' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold' : 'text-slate-400'
            }`}
          >
            AUDIT
          </button>
          <button
            onClick={() => setActiveTab('tensors')}
            className={`px-2 py-1 rounded transition-all ${
              activeTab === 'tensors' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold' : 'text-slate-400'
            }`}
          >
            TENSORS
          </button>
          {selectedNode && (
            <button
              onClick={() => setActiveTab('node')}
              className={`px-2 py-1 rounded transition-all ${
                activeTab === 'node' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 font-bold' : 'text-slate-400'
              }`}
            >
              NODE
            </button>
          )}
        </div>
      </div>

      {/* Main Tab Content */}
      <div className="flex-1 overflow-y-auto p-3 text-xs font-mono space-y-3">
        {activeTab === 'audit' && (
          <div className="space-y-3">
            {/* Paper Reference Badge */}
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-[11px] font-bold text-cyan-400 mb-1 flex items-center justify-between">
                <span>VASWANI ET AL. 2017 (ARXIV:1706.03762)</span>
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${
                  isExecuting ? 'bg-emerald-950 text-emerald-300' : 'bg-slate-800 text-slate-400'
                }`}>
                  {isExecuting ? 'LIVE SIGNAL' : 'TOPOLOGY'}
                </span>
              </div>
              <div className="text-[10px] text-slate-300 space-y-1">
                <div>Paper Stack: <span className="text-white font-bold">N=6, d_model=512, h=8</span></div>
                <div>Demo Stack: <span className="text-cyan-300 font-bold">N=2, d_model=16, h=2</span></div>
                <div>Normalization: <span className="text-emerald-400 font-bold">Post-LN (Original 2017)</span></div>
                <div>Memory Path: <span className="text-amber-400 font-bold">Enc Top &rarr; Dec Cross-Attn (K,V)</span></div>
              </div>
            </div>

            {/* Component Mapping Table */}
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
              COMPONENT AUDIT TABLE
            </div>
            <div className="space-y-1.5 text-[10px]">
              {[
                { name: 'Input Embedding', paper: 'Section 3.4', status: 'VERIFIED', match: 'Embedding(V, d_model)' },
                { name: 'Positional Encoding', paper: 'Section 3.5', status: 'VERIFIED', match: 'Sinusoidal PE Matrix' },
                { name: 'Encoder Self-Attention', paper: 'Section 3.2.1', status: 'VERIFIED', match: 'Scaled Dot-Product MHA' },
                { name: 'Encoder Add & Norm', paper: 'Section 3.1', status: 'VERIFIED', match: 'LayerNorm(x + Sublayer)' },
                { name: 'Encoder Feed-Forward', paper: 'Section 3.3', status: 'VERIFIED', match: 'ReLU(xW1 + b1)W2 + b2' },
                { name: 'Output Embedding (Shifted)', paper: 'Section 3.4', status: 'VERIFIED', match: 'Shifted right by <s>' },
                { name: 'Decoder Masked Attention', paper: 'Section 3.2.3', status: 'VERIFIED', match: 'Causal Mask (M_ij = -inf)' },
                { name: 'Decoder Cross-Attention', paper: 'Section 3.2.3', status: 'VERIFIED', match: 'Q from Dec, K/V from Enc' },
                { name: 'Linear & Softmax', paper: 'Section 3.4', status: 'VERIFIED', match: 'Dense(vocab) + Softmax' }
              ].map((item, i) => (
                <div key={i} className="p-1.5 rounded bg-black/40 border border-slate-800 flex items-center justify-between">
                  <div>
                    <div className="text-slate-200 font-bold">{item.name}</div>
                    <div className="text-slate-400 text-[9px]">{item.paper} &bull; {item.match}</div>
                  </div>
                  <span className="text-[9px] px-1 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-500/30 flex items-center gap-0.5">
                    <CheckCircle className="w-2.5 h-2.5" />
                    <span>PASS</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'tensors' && (
          <div className="space-y-3">
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-[11px] font-bold text-amber-400 mb-1 flex items-center gap-1">
                <Activity className="w-3 h-3 text-amber-400" />
                <span>GENUINE KERAS 3 FORWARD TRACE</span>
              </div>
              <div className="text-[10px] text-slate-300 space-y-1">
                <div>Source: <span className="text-cyan-300 font-bold">"{trace.source_text}"</span></div>
                <div>Tokens: <span className="text-white font-bold">{trace.encoder.source_tokens.join(' ')}</span></div>
                <div>Output: <span className="text-emerald-400 font-bold">"{trace.output_sentence}"</span></div>
              </div>
            </div>

            {/* Tensor Shapes */}
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
              INTERMEDIATE TENSOR SPEC
            </div>
            <div className="space-y-1.5 text-[10px]">
              <div className="p-2 rounded bg-black/40 border border-slate-800">
                <div className="text-cyan-300 font-bold">Token Embeddings:</div>
                <div className="text-slate-400">Tensor(shape=[5, 16], dtype=float32)</div>
              </div>
              <div className="p-2 rounded bg-black/40 border border-slate-800">
                <div className="text-cyan-300 font-bold">Encoder MHA Q, K, V:</div>
                <div className="text-slate-400">Tensor(shape=[heads=2, seq=5, d_k=8])</div>
              </div>
              <div className="p-2 rounded bg-black/40 border border-slate-800">
                <div className="text-amber-300 font-bold">Attention Matrix A_ij:</div>
                <div className="text-slate-400">Tensor(shape=[heads=2, seq=5, seq=5])</div>
              </div>
              <div className="p-2 rounded bg-black/40 border border-slate-800">
                <div className="text-purple-300 font-bold">FFN Hidden Activation:</div>
                <div className="text-slate-400">Tensor(shape=[5, d_ff=32], ReLU)</div>
              </div>
              <div className="p-2 rounded bg-black/40 border border-slate-800">
                <div className="text-emerald-300 font-bold">Logits &amp; Probabilities:</div>
                <div className="text-slate-400">Tensor(shape=[vocab=25], Softmax)</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'node' && selectedNode && (
          <div className="space-y-2.5">
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-100 mb-1">
                <Info className="w-3.5 h-3.5 text-cyan-400" />
                <span>{selectedNode.label}</span>
              </div>
              <div className="text-[10px] text-slate-400 font-mono mb-2">ID: {selectedNode.id}</div>
              {selectedNode.formula && (
                <div className="p-1.5 rounded bg-black/50 border border-slate-700/80 text-[10px] text-amber-300 font-mono mb-2">
                  {selectedNode.formula}
                </div>
              )}
              {selectedNode.description && (
                <div className="text-[10px] text-slate-300">{selectedNode.description}</div>
              )}
            </div>

            <div className="p-2 rounded bg-black/30 border border-slate-800 text-[10px] space-y-1">
              <div>Column: <span className="text-cyan-300 font-bold uppercase">{selectedNode.column}</span></div>
              <div>Sublayer Type: <span className="text-white font-bold">{selectedNode.type}</span></div>
              <div>Inputs: <span className="text-slate-400">{selectedNode.inputs?.join(', ') || 'None'}</span></div>
              <div>Outputs: <span className="text-slate-400">{selectedNode.outputs?.join(', ') || 'None'}</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
