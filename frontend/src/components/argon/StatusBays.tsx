import React from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw, Cpu, Radio, Zap, ShieldCheck } from 'lucide-react';

interface StatusBaysProps {
  assemblyStep: number;
  onSetAssemblyStep: (step: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  isExecuting: boolean;
  onToggleExecute: () => void;
  onResetCamera: () => void;
  activeHead: number;
  onChangeHead: (head: number) => void;
}

export const StatusBays: React.FC<StatusBaysProps> = ({
  assemblyStep,
  onSetAssemblyStep,
  isPlaying,
  onTogglePlay,
  isExecuting,
  onToggleExecute,
  onResetCamera,
  activeHead,
  onChangeHead
}) => {
  const STEPS = [
    { id: 0, label: '01. CHASSIS' },
    { id: 1, label: '02. EMBEDDING' },
    { id: 2, label: '03. POS-ENC' },
    { id: 3, label: '04. ENCODER BLOCK' },
    { id: 4, label: '05. PIPELINE READY' }
  ];

  return (
    <div className="w-full bg-[#080d19] border-t border-slate-800/90 p-3 flex flex-col gap-2.5 z-20 select-none shadow-2xl">
      {/* 1. TIMELINE SCRUBBER & CONTROL STRIP */}
      <div className="flex items-center justify-between gap-3 px-2">
        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onSetAssemblyStep(Math.max(0, assemblyStep - 1))}
            disabled={assemblyStep === 0}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 text-slate-200"
            title="Previous Assembly Step"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={onTogglePlay}
            className={`px-3 py-1.5 rounded-lg font-bold text-xs flex items-center gap-1.5 transition-all shadow-md ${
              isPlaying
                ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
            }`}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            <span>{isPlaying ? 'PAUSE BUILD' : 'AUTO BUILD'}</span>
          </button>

          <button
            onClick={() => onSetAssemblyStep(Math.min(4, assemblyStep + 1))}
            disabled={assemblyStep === 4}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 text-slate-200"
            title="Next Assembly Step"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>

          {/* Mode Action Button */}
          <button
            onClick={onToggleExecute}
            className={`ml-2 px-3 py-1.5 rounded-lg font-bold text-xs flex items-center gap-1.5 transition-all border ${
              isExecuting
                ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300 shadow-lg shadow-emerald-950/40'
                : 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300'
            }`}
          >
            <Zap className={`w-3.5 h-3.5 ${isExecuting ? 'text-emerald-400 fill-current' : 'text-slate-400'}`} />
            <span>{isExecuting ? 'EXECUTION RUNNING' : 'RUN KERAS TRACE'}</span>
          </button>
        </div>

        {/* Assembly Step Scrubber Pills */}
        <div className="hidden sm:flex items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
          {STEPS.map((s) => {
            const isActive = assemblyStep === s.id;
            const isCompleted = assemblyStep > s.id;
            return (
              <button
                key={s.id}
                onClick={() => onSetAssemblyStep(s.id)}
                className={`text-[11px] font-mono px-3 py-1 rounded-lg transition-all ${
                  isActive
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-950'
                    : isCompleted
                    ? 'text-cyan-400/80 hover:bg-slate-800'
                    : 'text-slate-500 hover:bg-slate-900'
                }`}
              >
                {s.label}
              </button>
            );
          })}
        </div>

        {/* Head Selector & Reset Camera */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-950 rounded-lg p-0.5 border border-slate-800 text-[11px] font-mono">
            <button
              onClick={() => onChangeHead(0)}
              className={`px-2 py-0.5 rounded ${activeHead === 0 ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400'}`}
            >
              H0
            </button>
            <button
              onClick={() => onChangeHead(1)}
              className={`px-2 py-0.5 rounded ${activeHead === 1 ? 'bg-rose-500 text-slate-950 font-bold' : 'text-slate-400'}`}
            >
              H1
            </button>
            <button
              onClick={() => onChangeHead(-1)}
              className={`px-2 py-0.5 rounded ${activeHead === -1 ? 'bg-purple-600 text-white font-bold' : 'text-slate-400'}`}
            >
              ALL
            </button>
          </div>

          <button
            onClick={onResetCamera}
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs border border-slate-700"
            title="Reset to Isometric View"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
            <span className="hidden md:inline">ISOMETRIC CAM</span>
          </button>
        </div>
      </div>

      {/* 2. FOUR MODULE-STATUS BAYS (ARGON REFERENCE SPEC) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
        {/* BAY 1: INPUT EMBEDDING */}
        <div className={`p-2.5 rounded-xl border transition-all ${
          assemblyStep >= 1
            ? 'bg-slate-900/80 border-cyan-500/40 shadow-sm shadow-cyan-950/20'
            : 'bg-slate-950/40 border-slate-800/40 opacity-50'
        }`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-bold text-slate-400 tracking-wider flex items-center gap-1">
              <Cpu className="w-3 h-3 text-cyan-400" />
              BAY 01: EMBEDDING
            </span>
            <span className={`text-[9px] px-1.5 py-0.2 rounded font-mono font-bold ${
              assemblyStep >= 1 ? 'bg-cyan-950 text-cyan-300' : 'bg-slate-800 text-slate-500'
            }`}>
              {assemblyStep >= 1 ? 'MOUNTED' : 'OFFLINE'}
            </span>
          </div>
          <div className="text-[11px] font-mono text-slate-200">Matrix: [5 x 8-D]</div>
          <div className="text-[10px] text-slate-400 font-mono">Bus: 8-Channel Vector</div>
        </div>

        {/* BAY 2: POSITIONAL RESONATOR */}
        <div className={`p-2.5 rounded-xl border transition-all ${
          assemblyStep >= 2
            ? 'bg-slate-900/80 border-amber-500/40 shadow-sm shadow-amber-950/20'
            : 'bg-slate-950/40 border-slate-800/40 opacity-50'
        }`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-bold text-slate-400 tracking-wider flex items-center gap-1">
              <Radio className="w-3 h-3 text-amber-400" />
              BAY 02: POS-ENC
            </span>
            <span className={`text-[9px] px-1.5 py-0.2 rounded font-mono font-bold ${
              assemblyStep >= 2 ? 'bg-amber-950 text-amber-300' : 'bg-slate-800 text-slate-500'
            }`}>
              {assemblyStep >= 2 ? 'COUPLED' : 'OFFLINE'}
            </span>
          </div>
          <div className="text-[11px] font-mono text-slate-200">Harmonics: Sin / Cos</div>
          <div className="text-[10px] text-slate-400 font-mono">Div Term: 10000^{'{2i/d}'}</div>
        </div>

        {/* BAY 3: ATTENTION CORE */}
        <div className={`p-2.5 rounded-xl border transition-all ${
          assemblyStep >= 3
            ? 'bg-slate-900/80 border-purple-500/40 shadow-sm shadow-purple-950/20'
            : 'bg-slate-950/40 border-slate-800/40 opacity-50'
        }`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-bold text-slate-400 tracking-wider flex items-center gap-1">
              <Zap className="w-3 h-3 text-purple-400" />
              BAY 03: ATTENTION
            </span>
            <span className={`text-[9px] px-1.5 py-0.2 rounded font-mono font-bold ${
              assemblyStep >= 3 ? 'bg-purple-950 text-purple-300' : 'bg-slate-800 text-slate-500'
            }`}>
              {assemblyStep >= 3 ? 'COUPLED' : 'OFFLINE'}
            </span>
          </div>
          <div className="text-[11px] font-mono text-slate-200">Core: 2 Heads (dk=4)</div>
          <div className="text-[10px] text-slate-400 font-mono">Formula: Softmax(QK^T/√d)V</div>
        </div>

        {/* BAY 4: FFN & RESIDUALS */}
        <div className={`p-2.5 rounded-xl border transition-all ${
          assemblyStep >= 3
            ? 'bg-slate-900/80 border-emerald-500/40 shadow-sm shadow-emerald-950/20'
            : 'bg-slate-950/40 border-slate-800/40 opacity-50'
        }`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-bold text-slate-400 tracking-wider flex items-center gap-1">
              <ShieldCheck className="w-3 h-3 text-emerald-400" />
              BAY 04: FFN &amp; LN
            </span>
            <span className={`text-[9px] px-1.5 py-0.2 rounded font-mono font-bold ${
              assemblyStep >= 3 ? 'bg-emerald-950 text-emerald-300' : 'bg-slate-800 text-slate-500'
            }`}>
              {assemblyStep >= 3 ? 'OPTIMAL' : 'OFFLINE'}
            </span>
          </div>
          <div className="text-[11px] font-mono text-slate-200">Expansion: 16-D ReLU</div>
          <div className="text-[10px] text-slate-400 font-mono">Norm: LayerNorm (ε=1e-6)</div>
        </div>
      </div>
    </div>
  );
};
