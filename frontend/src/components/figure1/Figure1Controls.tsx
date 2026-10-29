import React from 'react';
import { Eye, Zap, RotateCcw, Box, Split, ShieldAlert } from 'lucide-react';

interface Figure1ControlsProps {
  viewMode: 'global' | 'enc_exploded' | 'dec_exploded' | 'cross_focus' | 'causal_focus';
  onSetViewMode: (mode: 'global' | 'enc_exploded' | 'dec_exploded' | 'cross_focus' | 'causal_focus') => void;
  isExecuting: boolean;
  onToggleExecute: () => void;
  onResetCamera: () => void;
  sourceText: string;
  onSelectPreset: (text: string) => void;
}

export const Figure1Controls: React.FC<Figure1ControlsProps> = ({
  viewMode,
  onSetViewMode,
  isExecuting,
  onToggleExecute,
  onResetCamera,
  sourceText,
  onSelectPreset
}) => {
  const PRESETS = [
    { label: 'Sample Answer', text: 'Lắng nghe ý kiến nhóm giải quyết mâu thuẫn' },
    { label: 'Contrast Answer', text: 'Tôi biết python sql mô hình code' }
  ];

  const VIEW_OPTIONS = [
    { id: 'global', label: '1. Global Architecture', icon: Box },
    { id: 'enc_exploded', label: '2. Encoder Exploded', icon: Split },
    { id: 'dec_exploded', label: '3. Decoder Exploded', icon: Split },
    { id: 'cross_focus', label: '4. Cross-Attention Bus', icon: Eye },
    { id: 'causal_focus', label: '5. Causal Mask Shield', icon: ShieldAlert }
  ];

  return (
    <>
      {/* TOP SCIENTIFIC HEADER */}
      <header className="h-12 w-full bg-[#080d1a] border-b border-slate-800 px-4 flex items-center justify-between z-20 select-none shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded bg-blue-600 flex items-center justify-center text-white font-bold text-xs shadow-md">
            T17
          </div>
          <div>
            <div className="text-xs font-bold tracking-wider text-slate-100 uppercase font-mono">
              TRANSFORMER 2017 (VASWANI ET AL.) — FIGURE 1 3D VISUALIZATION
            </div>
          </div>
        </div>

        {/* View Mode Navigation Bar */}
        <div className="hidden md:flex items-center bg-slate-900 rounded-lg p-0.5 border border-slate-800 text-xs font-mono">
          {VIEW_OPTIONS.map((opt) => {
            const isActive = viewMode === opt.id;
            const Icon = opt.icon;
            return (
              <button
                key={opt.id}
                onClick={() => onSetViewMode(opt.id as any)}
                className={`px-3 py-1 rounded flex items-center gap-1.5 transition-all font-semibold ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{opt.label}</span>
              </button>
            );
          })}
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-2 font-mono text-xs">
          {PRESETS.map((p, idx) => {
            const isSelected = sourceText.toLowerCase().includes(p.text.toLowerCase().slice(0, 8));
            return (
              <button
                key={idx}
                onClick={() => onSelectPreset(p.text)}
                className={`px-2.5 py-1 rounded border transition-all ${
                  isSelected
                    ? 'bg-blue-950 border-blue-400 text-blue-200 font-bold'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* BOTTOM FLOATING ACTION BAR */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-xl px-4 py-2 shadow-2xl font-mono text-xs">
        <button
          onClick={onToggleExecute}
          className={`px-4 py-2 rounded-lg font-bold flex items-center gap-2 transition-all shadow-md ${
            isExecuting
              ? 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-emerald-950/50'
              : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-950/50'
          }`}
        >
          <Zap className={`w-4 h-4 ${isExecuting ? 'fill-current' : ''}`} />
          <span>{isExecuting ? 'STOP EXECUTION' : 'EXECUTE FORWARD PASS'}</span>
        </button>

        <div className="h-5 w-[1px] bg-slate-700" />

        <button
          onClick={onResetCamera}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
          title="Reset Camera"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
          <span>RESET CAM</span>
        </button>
      </div>
    </>
  );
};
