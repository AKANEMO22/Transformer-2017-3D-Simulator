import { useState, useEffect, useCallback } from 'react';
import { Figure1Scene } from './components/figure1/Figure1Scene';
import { Figure1Telemetry } from './components/figure1/Figure1Telemetry';
import { Figure1Controls } from './components/figure1/Figure1Controls';
import { fetchTrace, FALLBACK_TRACE } from './services/api';
import type { TransformerTrace } from './types/trace';

export default function App() {
  const [sourceText, setSourceText] = useState('Lắng nghe ý kiến nhóm');
  const [trace, setTrace] = useState<TransformerTrace>(FALLBACK_TRACE);
  const [loading, setLoading] = useState(false);

  // Figure 1 View Modes
  const [viewMode, setViewMode] = useState<'global' | 'enc_exploded' | 'dec_exploded' | 'cross_focus' | 'causal_focus'>('global');
  const [isExecuting, setIsExecuting] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [cameraResetTrigger, setCameraResetTrigger] = useState(0);

  // Load genuine Keras 3 trace
  const loadTrace = useCallback(async (text: string) => {
    setLoading(true);
    try {
      const data = await fetchTrace(text, 4);
      setTrace(data);
    } catch (err) {
      console.error('Failed to load trace:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTrace(sourceText);
  }, [loadTrace, sourceText]);

  const handleSelectPreset = (text: string) => {
    setSourceText(text);
    loadTrace(text);
  };

  return (
    <div className="w-screen h-screen flex flex-col bg-[#060a14] text-slate-100 overflow-hidden font-mono select-none">
      {/* 1. TOP HEADER & VIEW CONTROLS */}
      <Figure1Controls
        viewMode={viewMode}
        onSetViewMode={setViewMode}
        isExecuting={isExecuting}
        onToggleExecute={() => setIsExecuting(!isExecuting)}
        onResetCamera={() => setCameraResetTrigger((p) => p + 1)}
        sourceText={sourceText}
        onSelectPreset={handleSelectPreset}
      />

      {/* 2. MAIN WORKSPACE: LEFT 25% TELEMETRY + RIGHT 75% 3D SCENE */}
      <div className="flex-1 flex flex-row relative overflow-hidden">
        {/* Left 25%: Figure 1 Audit & Keras Tensor Stream */}
        <Figure1Telemetry
          trace={trace}
          selectedNodeId={selectedNodeId}
          isExecuting={isExecuting}
        />

        {/* Right 75%: Large Interactive Figure 1 3D Scene */}
        <div className="flex-1 h-full relative">
          <Figure1Scene
            trace={trace}
            viewMode={viewMode}
            isExecuting={isExecuting}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNodeId}
            resetTrigger={cameraResetTrigger}
          />

          {/* Loading Indicator */}
          {loading && (
            <div className="absolute top-4 right-4 z-30 flex items-center gap-2 bg-slate-900/90 border border-blue-500/50 rounded-lg px-3 py-1.5 shadow-xl">
              <div className="w-2.5 h-2.5 rounded-full border-2 border-blue-400 border-t-transparent animate-spin" />
              <span className="text-[11px] text-blue-300">Evaluating Keras 3 Architecture...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
