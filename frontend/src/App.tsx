import { useState, useEffect, useCallback, useRef } from 'react';
import { Figure1Scene } from './components/figure1/Figure1Scene';
import { Figure1Telemetry } from './components/figure1/Figure1Telemetry';
import { Figure1Controls } from './components/figure1/Figure1Controls';
import { executeTrace, ApiError } from './services/api';
import { generateDemoTrace } from './services/demoTrace';
import { buildExecutionTimeline } from './services/executionTimeline';
import type { TransformerTrace, TraceSource, TimelineStep } from './types/trace';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function App() {
  // Input & Simulation Mode
  const [inputText, setInputText] = useState('Lắng nghe ý kiến nhóm giải quyết mâu thuẫn');
  const [simulationMode, setSimulationMode] = useState<TraceSource>('simulated');
  const [activeTraceSource, setActiveTraceSource] = useState<TraceSource>('simulated');

  // Trace & Async State
  const [trace, setTrace] = useState<TransformerTrace | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<{ message: string; canSwitchToSim: boolean } | null>(null);
  const activeRunIdRef = useRef<string>('');

  // Timeline & Playback
  const [timeline, setTimeline] = useState<TimelineStep[]>([]);
  const [currentExecutionStep, setCurrentExecutionStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [activeDecoderStep, setActiveDecoderStep] = useState<number>(0);

  // 3D Scene View & Node Selection
  const [viewMode, setViewMode] = useState<'global' | 'enc_exploded' | 'dec_exploded' | 'cross_focus' | 'causal_focus'>('global');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [cameraResetTrigger, setCameraResetTrigger] = useState(0);

  // Generate trace explicitly
  const runSimulation = useCallback(async (textToRun: string, modeToUse: TraceSource) => {
    if (!textToRun.trim()) return;

    setIsLoading(true);
    setError(null);
    setIsPlaying(false);

    const runId = `${modeToUse}_${Date.now()}`;
    activeRunIdRef.current = runId;

    try {
      const result = await executeTrace(textToRun, modeToUse);
      // Ensure response matches current active run
      if (activeRunIdRef.current === runId) {
        setTrace(result);
        setActiveTraceSource(result.metadata?.source || modeToUse);

        // Build deterministic execution timeline
        const newTimeline = buildExecutionTimeline(result, 4);
        setTimeline(newTimeline);
        setCurrentExecutionStep(0);
        setActiveDecoderStep(0);
      }
    } catch (err: any) {
      if (activeRunIdRef.current === runId) {
        console.error('Simulation error:', err);
        const canSwitch = err instanceof ApiError ? err.canFallbackToSim : true;
        setError({
          message: err.message || 'Lỗi không xác định khi tạo trace.',
          canSwitchToSim: canSwitch && modeToUse === 'keras'
        });
      }
    } finally {
      if (activeRunIdRef.current === runId) {
        setIsLoading(false);
      }
    }
  }, []);

  // Bootstrapping: initialize initial educational demo trace on first load
  useEffect(() => {
    const initialTrace = generateDemoTrace('Lắng nghe ý kiến nhóm giải quyết mâu thuẫn');
    setTrace(initialTrace);
    setActiveTraceSource('simulated');
    const initialTimeline = buildExecutionTimeline(initialTrace, 4);
    setTimeline(initialTimeline);
    setCurrentExecutionStep(0);
  }, []);

  // Deterministic Timeline Playback Loop
  useEffect(() => {
    if (!isPlaying || timeline.length === 0) return;

    const currentStep = timeline[currentExecutionStep];
    const baseDuration = currentStep?.duration || 1200;
    const adjustedDuration = Math.max(300, baseDuration / playbackSpeed);

    const timer = setTimeout(() => {
      setCurrentExecutionStep((prev) => {
        if (prev + 1 >= timeline.length) {
          setIsPlaying(false);
          return prev;
        }
        const nextIdx = prev + 1;
        const nextStep = timeline[nextIdx];
        if (nextStep && nextStep.decoderStep !== undefined) {
          setActiveDecoderStep(nextStep.decoderStep);
        }
        return nextIdx;
      });
    }, adjustedDuration);

    return () => clearTimeout(timer);
  }, [isPlaying, currentExecutionStep, timeline, playbackSpeed]);

  // Timeline Controls
  const handlePlayToggle = () => {
    if (timeline.length === 0) return;
    if (currentExecutionStep >= timeline.length - 1) {
      setCurrentExecutionStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleNextStep = () => {
    setIsPlaying(false);
    if (currentExecutionStep < timeline.length - 1) {
      const nextIdx = currentExecutionStep + 1;
      setCurrentExecutionStep(nextIdx);
      const nextStep = timeline[nextIdx];
      if (nextStep && nextStep.decoderStep !== undefined) {
        setActiveDecoderStep(nextStep.decoderStep);
      }
    }
  };

  const handlePrevStep = () => {
    setIsPlaying(false);
    if (currentExecutionStep > 0) {
      const prevIdx = currentExecutionStep - 1;
      setCurrentExecutionStep(prevIdx);
      const prevStep = timeline[prevIdx];
      if (prevStep && prevStep.decoderStep !== undefined) {
        setActiveDecoderStep(prevStep.decoderStep);
      }
    }
  };

  const handleResetTimeline = () => {
    setIsPlaying(false);
    setCurrentExecutionStep(0);
    setActiveDecoderStep(0);
  };

  const handleSelectSample = (sampleText: string) => {
    setInputText(sampleText);
    runSimulation(sampleText, simulationMode);
  };

  const handleSwitchToSimulation = () => {
    setSimulationMode('simulated');
    setError(null);
    runSimulation(inputText, 'simulated');
  };

  // Active step info
  const activeTimelineStep = timeline[currentExecutionStep] || null;
  const activeNodeId = isPlaying ? activeTimelineStep?.nodeId : null;
  const activeEdgeIds = isPlaying ? activeTimelineStep?.activeEdgeIds || [] : [];

  return (
    <div className="w-screen h-screen flex flex-col bg-[#060a14] text-slate-100 overflow-hidden font-mono select-none">
      {/* 1. TOP CONTROLS & TIMELINE BAR */}
      <Figure1Controls
        inputText={inputText}
        onChangeInputText={setInputText}
        onSimulate={() => runSimulation(inputText, simulationMode)}
        isLoading={isLoading}
        simulationMode={simulationMode}
        onChangeSimulationMode={setSimulationMode}
        activeTraceSource={activeTraceSource}
        viewMode={viewMode}
        onSetViewMode={setViewMode}
        onResetCamera={() => setCameraResetTrigger((p) => p + 1)}
        isPlaying={isPlaying}
        onTogglePlay={handlePlayToggle}
        onPrevStep={handlePrevStep}
        onNextStep={handleNextStep}
        onResetTimeline={handleResetTimeline}
        playbackSpeed={playbackSpeed}
        onChangePlaybackSpeed={setPlaybackSpeed}
        currentStepIndex={currentExecutionStep}
        totalSteps={timeline.length || 18}
        currentStepLabel={activeTimelineStep?.label}
        activeDecoderStep={activeDecoderStep}
        maxDecoderSteps={trace?.decoder_steps?.length || 4}
        onSelectDecoderStep={(step) => {
          setActiveDecoderStep(step);
          // Jump timeline to first step of this decoder step
          const stepIndex = timeline.findIndex(
            (t) => t.decoderStep === step && t.nodeId === 'dec_outputs'
          );
          if (stepIndex >= 0) {
            setCurrentExecutionStep(stepIndex);
          }
        }}
        onSelectSample={handleSelectSample}
      />

      {/* ERROR MODAL / BANNER */}
      {error && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-50 max-w-xl w-full px-4">
          <div className="bg-red-950/95 border border-red-500/80 rounded-xl p-4 shadow-2xl backdrop-blur-md text-xs font-mono text-slate-200 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="flex-1 space-y-2">
              <div className="font-bold text-red-300">Không thể kết nối Backend Keras</div>
              <p className="text-slate-300 leading-relaxed font-sans">{error.message}</p>
              {error.canSwitchToSim && (
                <div className="pt-1 flex items-center gap-2">
                  <button
                    onClick={handleSwitchToSimulation}
                    className="px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 shadow-md shadow-cyan-950/50 cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Chuyển sang Educational Simulation</span>
                  </button>
                  <button
                    onClick={() => setError(null)}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
                  >
                    Đóng
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 2. MAIN WORKSPACE: LEFT TELEMETRY/INSPECTOR + RIGHT 3D SCENE */}
      <div className="flex-1 flex flex-row relative overflow-hidden">
        {/* Left Telemetry & Node Inspector */}
        <Figure1Telemetry
          trace={trace}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
          isExecuting={isPlaying}
          activeDecoderStep={activeDecoderStep}
          currentExecutionStep={currentExecutionStep}
          totalExecutionSteps={timeline.length || 18}
        />

        {/* Right Large Interactive 3D Scene */}
        <div className="flex-1 h-full relative">
          <Figure1Scene
            trace={trace}
            viewMode={viewMode}
            isExecuting={isPlaying}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNodeId}
            activeNodeId={activeNodeId}
            activeEdgeIds={activeEdgeIds}
            activeDecoderStep={activeDecoderStep}
            resetTrigger={cameraResetTrigger}
          />
        </div>
      </div>
    </div>
  );
}
