import React from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Zap,
  Box,
  Split,
  Eye,
  ShieldAlert,
  Loader2,
  Cpu
} from 'lucide-react';
import type { TraceSource } from '../../types/trace';

interface Figure1ControlsProps {
  inputText: string;
  onChangeInputText: (val: string) => void;
  onSimulate: () => void;
  isLoading: boolean;
  simulationMode: TraceSource;
  onChangeSimulationMode: (mode: TraceSource) => void;
  activeTraceSource?: TraceSource;
  // View mode
  viewMode: 'global' | 'enc_exploded' | 'dec_exploded' | 'cross_focus' | 'causal_focus';
  onSetViewMode: (mode: 'global' | 'enc_exploded' | 'dec_exploded' | 'cross_focus' | 'causal_focus') => void;
  onResetCamera: () => void;
  // Timeline playback
  isPlaying: boolean;
  onTogglePlay: () => void;
  onPrevStep: () => void;
  onNextStep: () => void;
  onResetTimeline: () => void;
  playbackSpeed: number;
  onChangePlaybackSpeed: (speed: number) => void;
  // Execution step details
  currentStepIndex: number;
  totalSteps: number;
  currentStepLabel?: string;
  // Decoder step selector
  activeDecoderStep: number;
  maxDecoderSteps: number;
  onSelectDecoderStep: (step: number) => void;
  // Presets
  onSelectSample: (text: string) => void;
  // Mode switch (Argon vs Figure 1)
  onToggleAppMode?: () => void;
}

export const Figure1Controls: React.FC<Figure1ControlsProps> = ({
  inputText,
  onChangeInputText,
  onSimulate,
  isLoading,
  simulationMode,
  onChangeSimulationMode,
  activeTraceSource,
  viewMode,
  onSetViewMode,
  onResetCamera,
  isPlaying,
  onTogglePlay,
  onPrevStep,
  onNextStep,
  onResetTimeline,
  playbackSpeed,
  onChangePlaybackSpeed,
  currentStepIndex,
  totalSteps,
  currentStepLabel,
  activeDecoderStep,
  maxDecoderSteps,
  onSelectDecoderStep,
  onSelectSample,
  onToggleAppMode
}) => {
  const SAMPLES = [
    { label: 'Phỏng vấn AI', text: 'Lắng nghe ý kiến nhóm giải quyết mâu thuẫn' },
    { label: 'Lạc đề đối chứng', text: 'Tôi biết python sql mô hình code' },
    { label: '2017 Nguyên bản', text: 'Ý kiến thảo luận giải quyết đồng thuận' }
  ];

  const VIEW_OPTIONS = [
    { id: 'global', label: 'Figure 1 Overview', icon: Box },
    { id: 'enc_exploded', label: 'Enc Exploded', icon: Split },
    { id: 'dec_exploded', label: 'Dec Exploded', icon: Split },
    { id: 'cross_focus', label: 'Cross-Attn Bus', icon: Eye },
    { id: 'causal_focus', label: 'Causal Shield', icon: ShieldAlert }
  ];

  const isSimulateDisabled = !inputText.trim() || isLoading;

  return (
    <div className="z-20 w-full flex flex-col font-mono select-none">
      {/* TOP HEADER: BRAND + INPUT BAR + SIMULATE + PROVENANCE BADGE */}
      <header className="w-full bg-[#080d1a]/95 border-b border-slate-800/80 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 shadow-xl backdrop-blur-md">
        {/* Left: Brand + App Mode Switcher */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-xs shadow-lg shadow-cyan-500/20">
              T17
            </div>
            <div>
              <div className="text-xs font-bold tracking-wider text-slate-100 flex items-center gap-1.5">
                <span>TRANSFORMER 2017</span>
                <span className="text-[10px] text-cyan-400 font-normal border border-cyan-800/60 px-1 py-0.2 rounded bg-cyan-950/60">
                  FIGURE 1 SIMULATOR
                </span>
              </div>
            </div>
          </div>

          {onToggleAppMode && (
            <button
              onClick={onToggleAppMode}
              className="text-[10px] px-2 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors flex items-center gap-1"
              title="Chuyển sang giao diện phần cứng ARGON-TX7"
            >
              <Cpu className="w-3 h-3 text-cyan-400" />
              <span>ARGON MODE</span>
            </button>
          )}
        </div>

        {/* Center: Input Sentence + Mode Selector + SIMULATE Button */}
        <div className="flex-1 max-w-2xl flex items-center gap-2">
          {/* Text Input */}
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputText}
              onChange={(e) => onChangeInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !isSimulateDisabled) {
                  onSimulate();
                }
              }}
              placeholder="Nhập câu để xem Transformer xử lý (ví dụ: Lắng nghe ý kiến nhóm)..."
              maxLength={80}
              className="w-full bg-slate-950 border border-slate-700/80 focus:border-cyan-400 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 outline-none transition-all shadow-inner font-sans"
            />
          </div>

          {/* Mode Selector */}
          <select
            value={simulationMode}
            onChange={(e) => onChangeSimulationMode(e.target.value as TraceSource)}
            className="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 outline-none font-mono cursor-pointer hover:border-slate-600 transition-all"
            title="Chọn nguồn dữ liệu: Keras Backend thật hoặc Mô phỏng toán học cục bộ"
          >
            <option value="simulated">Mô phỏng (Educational Simulation)</option>
            <option value="keras">Keras 3 Backend (Real Forward Pass)</option>
          </select>

          {/* SIMULATE Button */}
          <button
            onClick={onSimulate}
            disabled={isSimulateDisabled}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all shadow-lg ${
              isSimulateDisabled
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/25 border border-cyan-400 cursor-pointer active:scale-95'
            }`}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Đang tính...</span>
              </>
            ) : (
              <>
                <Zap className="w-3.5 h-3.5 fill-current" />
                <span>SIMULATE</span>
              </>
            )}
          </button>
        </div>

        {/* Right: Actual Trace Provenance Badge */}
        <div className="flex items-center gap-2 text-[11px]">
          <div
            className={`px-2.5 py-1 rounded-md border font-semibold flex items-center gap-1.5 ${
              activeTraceSource === 'keras'
                ? 'bg-emerald-950/80 border-emerald-500/50 text-emerald-300 shadow-sm shadow-emerald-900/30'
                : 'bg-cyan-950/80 border-cyan-500/50 text-cyan-300 shadow-sm shadow-cyan-900/30'
            }`}
            title="Nguồn gốc thực tế của dữ liệu đang hiển thị"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                activeTraceSource === 'keras' ? 'bg-emerald-400' : 'bg-cyan-400'
              }`}
            />
            <span>
              {activeTraceSource === 'keras' ? 'REAL KERAS TRACE' : 'EDUCATIONAL SIMULATION'}
            </span>
          </div>
        </div>
      </header>

      {/* SECONDARY ROW: SAMPLES + 3D VIEW MODES */}
      <div className="w-full bg-[#0a0f1f]/80 border-b border-slate-800/60 px-4 py-1.5 flex flex-wrap items-center justify-between text-xs gap-2 backdrop-blur-sm">
        {/* Sample Presets */}
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-slate-500 uppercase mr-1">Mẫu thử:</span>
          {SAMPLES.map((s, idx) => (
            <button
              key={idx}
              onClick={() => onSelectSample(s.text)}
              className="text-[11px] px-2 py-0.5 rounded bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-cyan-300 transition-colors"
            >
              {s.label}
            </button>
          ))}
        </div>

        {/* 3D Camera / Sublayer Views */}
        <div className="flex items-center gap-1">
          {VIEW_OPTIONS.map((opt) => {
            const isActive = viewMode === opt.id;
            const Icon = opt.icon;
            return (
              <button
                key={opt.id}
                onClick={() => onSetViewMode(opt.id as any)}
                className={`text-[11px] px-2.5 py-1 rounded flex items-center gap-1.5 transition-all ${
                  isActive
                    ? 'bg-cyan-950 border border-cyan-500/60 text-cyan-300 font-bold shadow-sm'
                    : 'bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3 h-3" />
                <span>{opt.label}</span>
              </button>
            );
          })}
          <button
            onClick={onResetCamera}
            className="text-[11px] px-2 py-1 rounded bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-white transition-colors ml-1"
            title="Đưa góc nhìn về phối cảnh chuẩn"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* BOTTOM TIMELINE CONTROLLER (DETERMINISTIC PLAYBACK) */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-slate-950/95 backdrop-blur-xl border border-slate-700/80 rounded-2xl px-5 py-2.5 shadow-2xl font-mono text-xs max-w-4xl w-[94%] justify-between">
        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={onResetTimeline}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition-colors"
            title="Về đầu timeline"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={onPrevStep}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 transition-colors"
            title="Bước trước"
          >
            <SkipBack className="w-4 h-4" />
          </button>
          <button
            onClick={onTogglePlay}
            className={`px-3.5 py-1.5 rounded-lg font-bold flex items-center gap-1.5 transition-all shadow-md ${
              isPlaying
                ? 'bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-amber-500/20'
                : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/20'
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="w-4 h-4 fill-current" />
                <span>PAUSE</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>PLAY</span>
              </>
            )}
          </button>
          <button
            onClick={onNextStep}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 transition-colors"
            title="Bước tiếp theo"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Speed selector */}
          <div className="flex items-center gap-1 ml-2 text-[10px] text-slate-400 border-l border-slate-800 pl-2">
            <span>Tốc độ:</span>
            {[0.5, 1, 2].map((s) => (
              <button
                key={s}
                onClick={() => onChangePlaybackSpeed(s)}
                className={`px-1.5 py-0.5 rounded ${
                  playbackSpeed === s
                    ? 'bg-cyan-500 text-black font-bold'
                    : 'bg-slate-900 text-slate-400 hover:text-white'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Current Execution Step Indicator */}
        <div className="flex-1 mx-4 text-center truncate">
          <div className="text-[10px] text-slate-400 flex items-center justify-center gap-2">
            <span>Tiến trình:</span>
            <span className="text-cyan-400 font-bold">
              Bước {currentStepIndex + 1} / {totalSteps}
            </span>
          </div>
          <div className="text-xs text-white font-semibold truncate mt-0.5">
            {currentStepLabel || 'Chưa chạy'}
          </div>
        </div>

        {/* Decoder Step Selector */}
        <div className="flex items-center gap-1.5 text-[11px] border-l border-slate-800 pl-3">
          <span className="text-slate-400 text-[10px]">Decode Step:</span>
          <div className="flex gap-1">
            {Array.from({ length: maxDecoderSteps }, (_, i) => (
              <button
                key={i}
                onClick={() => onSelectDecoderStep(i)}
                className={`w-6 h-6 rounded flex items-center justify-center text-[10px] font-bold transition-all ${
                  activeDecoderStep === i
                    ? 'bg-amber-500 text-black shadow-md shadow-amber-500/30'
                    : 'bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-800'
                }`}
                title={`Xem bước sinh từ số ${i}`}
              >
                {i}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
