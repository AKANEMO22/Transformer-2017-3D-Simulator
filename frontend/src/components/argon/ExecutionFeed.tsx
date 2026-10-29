import React, { useState } from 'react';
import { Terminal, Code, Cpu, Activity, Database, CheckCircle2 } from 'lucide-react';
import type { TransformerTrace } from '../../types/trace';

interface ExecutionFeedProps {
  trace: TransformerTrace;
  assemblyStep: number;
  isExecuting: boolean;
  activeHead: number;
}

export const ExecutionFeed: React.FC<ExecutionFeedProps> = ({
  trace,
  assemblyStep,
  isExecuting,
  activeHead
}) => {
  const [activeTab, setActiveTab] = useState<'telemetry' | 'code'>('telemetry');

  const tokens = trace.encoder.source_tokens;

  // Real Keras 3 Code Snippets
  const kerasCodeSnippet = `# Keras 3 Transformer 2017 Architecture
import keras
from keras import layers, ops

class EncoderBlock(layers.Layer):
    def __init__(self, d_model=8, num_heads=2, d_ff=16):
        super().__init__()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model)
        ])
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        # Multi-Head Attention + Residual
        attn_out = self.mha(x, x)
        x_norm1 = self.ln1(x + attn_out)
        
        # Feed-Forward + Residual
        ffn_out = self.ffn(x_norm1)
        return self.ln2(x_norm1 + ffn_out)
`;

  return (
    <div className="w-[320px] lg:w-[360px] h-full flex flex-col bg-[#070b14] border-r border-slate-800/80 shadow-2xl z-10 select-none">
      {/* Feed Header */}
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-[#0a0f1d]">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold tracking-wider text-slate-100 uppercase">
            BUS TELEMETRY
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setActiveTab('telemetry')}
            className={`text-[11px] px-2.5 py-1 rounded transition-all font-mono ${
              activeTab === 'telemetry'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            FEED
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`text-[11px] px-2.5 py-1 rounded transition-all font-mono flex items-center gap-1 ${
              activeTab === 'code'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 font-bold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Code className="w-3 h-3" />
            <span>KERAS</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-3 text-xs font-mono space-y-3">
        {activeTab === 'telemetry' ? (
          <>
            {/* System Status Card */}
            <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
              <div className="flex items-center justify-between mb-1.5 text-[11px]">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  PIPELINE STATE:
                </span>
                <span className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
                  isExecuting
                    ? 'bg-emerald-950 text-emerald-400 border border-emerald-500/30'
                    : 'bg-cyan-950 text-cyan-400 border border-cyan-500/30'
                }`}>
                  {isExecuting ? 'LIVE EXECUTION' : 'ASSEMBLY MODE'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-300">
                <div>Backend: <span className="text-cyan-300 font-bold">Keras 3 + TF</span></div>
                <div>d_model: <span className="text-cyan-300 font-bold">{trace.d_model} (8-D)</span></div>
                <div>Heads: <span className="text-cyan-300 font-bold">{trace.num_heads} (dk=4)</span></div>
                <div>Active Head: <span className="text-amber-300 font-bold">{activeHead === -1 ? 'ALL' : activeHead}</span></div>
              </div>
            </div>

            {/* Live Streaming Log Events */}
            <div className="space-y-1.5 text-[11px]">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">
                EXECUTION EVENT LOG
              </div>

              <div className="p-2 rounded bg-black/40 border border-slate-800/80 space-y-1 text-slate-300">
                <div className="text-cyan-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                  <span>[00:01.05] CHASSIS_INITIALIZED</span>
                </div>
                <div className="text-slate-400 text-[10px] pl-4">
                  Mounting socket 0 (Embedding) &amp; socket 1 (Encoder)
                </div>

                {assemblyStep >= 1 && (
                  <div className="text-cyan-300 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-cyan-300" />
                    <span>[00:01.20] MOD_01_EMBEDDING_DOCKED</span>
                  </div>
                )}
                {assemblyStep >= 2 && (
                  <div className="text-amber-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-amber-400" />
                    <span>[00:01.35] MOD_02_POS_ENC_COUPLED</span>
                  </div>
                )}
                {assemblyStep >= 3 && (
                  <div className="text-purple-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-purple-400" />
                    <span>[00:01.50] MOD_03_ENCODER_LOCKED</span>
                  </div>
                )}

                {isExecuting && (
                  <>
                    <div className="text-emerald-400 animate-pulse flex items-center gap-1">
                      <Cpu className="w-3 h-3 text-emerald-400" />
                      <span>[EXEC] Q, K, V Splitting Heads &rarr; [2, 5, 4]</span>
                    </div>
                    <div className="text-cyan-300 pl-4 text-[10px]">
                      Attention(Q,K,V) = softmax(QK^T / 2.0) V
                    </div>
                    <div className="text-purple-300 pl-4 text-[10px]">
                      FFN expanding 8-D to 16-D ReLU &rarr; 8-D
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Token Channel Matrix */}
            <div className="p-2.5 rounded-lg bg-slate-900/70 border border-slate-800">
              <div className="flex items-center gap-1.5 text-[11px] font-bold text-slate-300 mb-2">
                <Database className="w-3.5 h-3.5 text-cyan-400" />
                <span>CHANNEL TOKENS ({tokens.length})</span>
              </div>
              <div className="space-y-1">
                {tokens.map((tok, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-1 px-2 rounded bg-black/30 border border-slate-800/60 text-[11px]"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500 font-mono text-[10px]">CH{i}</span>
                      <span className="font-bold text-cyan-200">{tok}</span>
                    </div>
                    <span className="text-[10px] text-slate-400">ID:{trace.encoder.source_ids[i] ?? (i + 3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          /* Real Keras Code View */
          <div className="p-2 bg-black/60 rounded-lg border border-slate-800 text-[11px] overflow-x-auto text-slate-300">
            <pre className="text-slate-300 whitespace-pre-wrap leading-relaxed">
              {kerasCodeSnippet}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
