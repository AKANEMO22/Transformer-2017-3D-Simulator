import React, { useState } from 'react';
import { 
  Play, Pause, SkipBack, SkipForward, RotateCcw, 
  Layers, Cpu, Sparkles, MessageSquare
} from 'lucide-react';
import { STAGES } from './ThreeCanvas';
import type { StageInfo } from '../types/trace';

interface HUDProps {
  currentStageIndex: number;
  onSelectStage: (idx: number) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  playbackSpeed: number;
  onChangeSpeed: (spd: number) => void;
  activeHead: number;
  onChangeHead: (head: number) => void;
  onResetCamera: () => void;
  sourceText: string;
  onSelectPreset: (text: string) => void;
  onSubmitCustomText: (text: string) => void;
  totalDecoderSteps: number;
  activeDecoderStep: number;
  onChangeDecoderStep: (step: number) => void;
  isCustomModalOpen: boolean;
  setIsCustomModalOpen: (open: boolean) => void;
}

export const HUD: React.FC<HUDProps> = ({
  currentStageIndex,
  onSelectStage,
  isPlaying,
  onTogglePlay,
  playbackSpeed,
  onChangeSpeed,
  activeHead,
  onChangeHead,
  onResetCamera,
  sourceText,
  onSelectPreset,
  onSubmitCustomText,
  totalDecoderSteps,
  activeDecoderStep,
  onChangeDecoderStep,
  isCustomModalOpen,
  setIsCustomModalOpen
}) => {
  const [customInput, setCustomInput] = useState(sourceText);
  const currentStage: StageInfo = STAGES[currentStageIndex] || STAGES[0];

  const PRESETS = [
    {
      id: 'interview_sample',
      label: 'AI Interview (Mẫu)',
      text: 'Lắng nghe ý kiến nhóm giải quyết mâu thuẫn',
      badge: 'Trọng tâm'
    },
    {
      id: 'interview_contrast',
      label: 'AI Interview (Đối chứng)',
      text: 'Tôi biết python sql mô hình code',
      badge: 'Lạc đề'
    }
  ];

  const isDecoderStage = ['decoder_masked', 'cross_attn', 'softmax'].includes(currentStage.id);

  return (
    <>
      {/* TOP FLOATING HEADER HUD */}
      <div className="absolute top-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
        {/* Left: Branding & Formula Badge */}
        <div className="flex items-center gap-3 bg-slate-900/80 backdrop-blur-md border border-cyan-500/30 rounded-xl px-4 py-2.5 shadow-lg shadow-cyan-950/40 pointer-events-auto">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-slate-950 font-black shadow-md">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold tracking-wider text-cyan-400 uppercase">
                Transformer 2017
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-500/30 text-cyan-300 font-mono">
                3D Visual Simulator
              </span>
            </div>
            <div className="text-sm font-semibold text-slate-100 flex items-center gap-1.5">
              <span>{currentStage.name}</span>
            </div>
          </div>
        </div>

        {/* Center: Live Mathematical Grounding Pill */}
        <div className="hidden lg:flex items-center gap-2 bg-slate-900/80 backdrop-blur-md border border-slate-700/60 rounded-full px-5 py-2 shadow-md pointer-events-auto">
          <span className="text-xs text-slate-400 font-mono">Formula:</span>
          <span className="text-xs font-mono font-medium text-amber-300">
            {currentStage.formula}
          </span>
        </div>

        {/* Right: Preset Prompts & Keras Grounded Badge */}
        <div className="flex items-center gap-2 pointer-events-auto">
          {PRESETS.map((p) => {
            const isSelected = sourceText.toLowerCase().includes(p.text.toLowerCase().slice(0, 10));
            return (
              <button
                key={p.id}
                onClick={() => onSelectPreset(p.text)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 backdrop-blur-md shadow-sm ${
                  isSelected
                    ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-cyan-900/30 font-semibold'
                    : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:border-slate-500 hover:text-white'
                }`}
              >
                <MessageSquare className="w-3.5 h-3.5 text-cyan-400" />
                <span>{p.label}</span>
              </button>
            );
          })}

          <button
            onClick={() => setIsCustomModalOpen(true)}
            className="text-xs px-3 py-1.5 rounded-lg border border-purple-500/40 bg-purple-950/40 hover:bg-purple-900/60 text-purple-200 transition-all backdrop-blur-md flex items-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span>Custom Text</span>
          </button>
        </div>
      </div>

      {/* DECODER STEPPER (APPEARS DURING DECODER STAGES 6, 7, 8) */}
      {isDecoderStage && totalDecoderSteps > 1 && (
        <div className="absolute top-20 right-4 z-20 flex items-center gap-1.5 bg-slate-900/85 backdrop-blur-md border border-purple-500/40 rounded-xl px-3 py-2 shadow-xl">
          <span className="text-[11px] font-bold text-purple-300 uppercase tracking-wide mr-1">
            Autoregressive Step:
          </span>
          {Array.from({ length: totalDecoderSteps }).map((_, stepIdx) => (
            <button
              key={stepIdx}
              onClick={() => onChangeDecoderStep(stepIdx)}
              className={`text-xs px-2.5 py-1 rounded font-mono font-medium transition-all ${
                activeDecoderStep === stepIdx
                  ? 'bg-purple-600 text-white shadow-md shadow-purple-900/50 scale-105'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
              }`}
            >
              t={stepIdx}
            </button>
          ))}
        </div>
      )}

      {/* BOTTOM FLOATING CONTROLS HUD */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20 w-[94%] max-w-5xl flex flex-col gap-2.5">
        {/* Stage Progress Pills */}
        <div className="flex items-center justify-between gap-1 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-xl p-1.5 shadow-xl overflow-x-auto">
          {STAGES.map((s, idx) => {
            const isActive = currentStageIndex === idx;
            const isPassed = currentStageIndex > idx;

            return (
              <button
                key={s.id}
                onClick={() => onSelectStage(idx)}
                className={`flex-1 min-w-[75px] py-1.5 px-2 rounded-lg text-xs font-medium transition-all flex flex-col items-center gap-0.5 ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold shadow-md shadow-cyan-950 scale-[1.02]'
                    : isPassed
                    ? 'text-cyan-400/80 hover:bg-slate-800/80 hover:text-cyan-300'
                    : 'text-slate-500 hover:bg-slate-800/60 hover:text-slate-300'
                }`}
              >
                <span className="text-[10px] opacity-75 font-mono">{idx + 1}</span>
                <span className="truncate w-full text-center">{s.shortName}</span>
              </button>
            );
          })}
        </div>

        {/* Playback & Control Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/90 backdrop-blur-lg border border-slate-700/60 rounded-2xl px-5 py-3 shadow-2xl">
          {/* Playback Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => onSelectStage(Math.max(0, currentStageIndex - 1))}
              disabled={currentStageIndex === 0}
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:pointer-events-none text-slate-200 transition-all"
              title="Previous Stage"
            >
              <SkipBack className="w-4 h-4" />
            </button>

            <button
              onClick={onTogglePlay}
              className={`px-4 py-2 rounded-xl font-semibold text-xs flex items-center gap-2 transition-all shadow-lg ${
                isPlaying
                  ? 'bg-amber-500 hover:bg-amber-600 text-slate-950 shadow-amber-950/50'
                  : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-950/50'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
              <span>{isPlaying ? 'PAUSE' : 'PLAY SIMULATION'}</span>
            </button>

            <button
              onClick={() => onSelectStage(Math.min(STAGES.length - 1, currentStageIndex + 1))}
              disabled={currentStageIndex === STAGES.length - 1}
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:pointer-events-none text-slate-200 transition-all"
              title="Next Stage"
            >
              <SkipForward className="w-4 h-4" />
            </button>

            {/* Speed Toggle */}
            <div className="flex items-center bg-slate-800 rounded-lg p-0.5 ml-2 border border-slate-700">
              {[0.5, 1, 2].map((spd) => (
                <button
                  key={spd}
                  onClick={() => onChangeSpeed(spd)}
                  className={`text-[11px] px-2 py-1 rounded font-mono font-medium transition-all ${
                    playbackSpeed === spd
                      ? 'bg-cyan-500 text-slate-950 font-bold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {spd}x
                </button>
              ))}
            </div>
          </div>

          {/* Attention Head Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 flex items-center gap-1 font-medium">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Attn Head:</span>
            </span>
            <div className="flex items-center bg-slate-800 rounded-lg p-0.5 border border-slate-700">
              <button
                onClick={() => onChangeHead(0)}
                className={`text-xs px-2.5 py-1 rounded transition-all font-medium ${
                  activeHead === 0
                    ? 'bg-cyan-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Head 1 (Cyan)
              </button>
              <button
                onClick={() => onChangeHead(1)}
                className={`text-xs px-2.5 py-1 rounded transition-all font-medium ${
                  activeHead === 1
                    ? 'bg-amber-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Head 2 (Amber)
              </button>
              <button
                onClick={() => onChangeHead(-1)}
                className={`text-xs px-2.5 py-1 rounded transition-all font-medium ${
                  activeHead === -1
                    ? 'bg-purple-600 text-white font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                All Heads
              </button>
            </div>
          </div>

          {/* Reset Camera Button */}
          <button
            onClick={onResetCamera}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs border border-slate-700 transition-all"
            title="Reset 3D Camera"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
            <span>Reset View</span>
          </button>
        </div>
      </div>

      {/* CUSTOM TEXT MODAL */}
      {isCustomModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg bg-slate-900 border border-cyan-500/40 rounded-2xl p-6 shadow-2xl shadow-cyan-950/60">
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2 mb-2">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              <span>Nhập câu văn để mô phỏng Transformer 2017</span>
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Nhập câu văn tiếng Việt hoặc tiếng Anh. Hệ thống Keras 3 sẽ tính toán toàn bộ forward-pass (Embedding, Positional Encoding, Multi-Head Attention, Masked Causal, Cross-Attention) theo thời gian thực.
            </p>

            <textarea
              value={customInput}
              onChange={(e) => setCustomInput(e.target.value)}
              rows={3}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl p-3 text-sm text-slate-100 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 mb-4 resize-none"
              placeholder="Nhập câu trả lời hoặc câu văn..."
            />

            <div className="flex items-center justify-end gap-2.5">
              <button
                onClick={() => setIsCustomModalOpen(false)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-all"
              >
                Đóng
              </button>
              <button
                onClick={() => {
                  onSubmitCustomText(customInput);
                  setIsCustomModalOpen(false);
                }}
                className="px-5 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 shadow-lg shadow-cyan-950/50 transition-all"
              >
                Mô phỏng 3D ngay
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
