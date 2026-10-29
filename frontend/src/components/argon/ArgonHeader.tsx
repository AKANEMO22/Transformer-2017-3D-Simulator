import React from 'react';
import { Cpu, Hammer, Zap, RefreshCw } from 'lucide-react';

interface ArgonHeaderProps {
  mode: 'assembly' | 'execution';
  onSetMode: (m: 'assembly' | 'execution') => void;
  assemblyStep: number;
  sourceText: string;
  onSelectPreset: (text: string) => void;
  onReloadTrace: () => void;
}

export const ArgonHeader: React.FC<ArgonHeaderProps> = ({
  mode,
  onSetMode,
  assemblyStep,
  sourceText,
  onSelectPreset,
  onReloadTrace
}) => {
  const PRESETS = [
    { label: 'Interview (Mẫu)', text: 'Lắng nghe ý kiến nhóm giải quyết mâu thuẫn' },
    { label: 'Technical (Đối chứng)', text: 'Tôi biết python sql mô hình code' }
  ];

  const assemblyPercent = Math.min(100, Math.round((assemblyStep / 3) * 100));

  return (
    <header className="h-12 w-full bg-[#060912] border-b border-slate-800/80 px-4 flex items-center justify-between z-20 select-none shadow-md">
      {/* Left: System Identifier */}
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 rounded bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-slate-950 font-black shadow-md shadow-cyan-950">
          <Cpu className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-black tracking-widest text-cyan-400 font-mono">
              ARGON-TX7
            </span>
            <span className="text-[10px] text-slate-500 font-mono">
              // TRANSFORMER 2017 3D ASSEMBLY
            </span>
          </div>
        </div>
      </div>

      {/* Center: Mode Switcher & Progress */}
      <div className="flex items-center gap-3 font-mono">
        <div className="flex items-center bg-slate-900 rounded-lg p-0.5 border border-slate-800 text-xs">
          <button
            onClick={() => onSetMode('assembly')}
            className={`px-3 py-1 rounded flex items-center gap-1.5 font-bold transition-all ${
              mode === 'assembly'
                ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Hammer className="w-3.5 h-3.5" />
            <span>MODE 1: BUILD</span>
          </button>
          <button
            onClick={() => onSetMode('execution')}
            className={`px-3 py-1 rounded flex items-center gap-1.5 font-bold transition-all ${
              mode === 'execution'
                ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-950'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Zap className="w-3.5 h-3.5 fill-current" />
            <span>MODE 2: EXECUTE</span>
          </button>
        </div>

        {/* Assembly Progress Meter */}
        <div className="hidden md:flex items-center gap-2 bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800 text-[11px]">
          <span className="text-slate-500">BUILD:</span>
          <span className="text-cyan-300 font-bold">{assemblyPercent}%</span>
          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-cyan-400 transition-all duration-300"
              style={{ width: `${assemblyPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Right: Presets & Grounded Status */}
      <div className="flex items-center gap-2">
        {PRESETS.map((p, idx) => {
          const isSelected = sourceText.toLowerCase().includes(p.text.toLowerCase().slice(0, 8));
          return (
            <button
              key={idx}
              onClick={() => onSelectPreset(p.text)}
              className={`text-[11px] font-mono px-2.5 py-1 rounded border transition-all ${
                isSelected
                  ? 'bg-cyan-950 border-cyan-400 text-cyan-200 font-bold'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {p.label}
            </button>
          );
        })}

        <button
          onClick={onReloadTrace}
          className="p-1 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-cyan-400"
          title="Re-run Keras 3 Trace"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>
    </header>
  );
};
