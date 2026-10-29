import React, { useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import type { TransformerTrace } from '../../types/trace';

import { BasePlatform } from './BasePlatform';
import { EmbeddingModule } from './EmbeddingModule';
import { PositionalModule } from './PositionalModule';
import { EncoderBlockModule } from './EncoderBlockModule';
import { SignalBus } from './SignalBus';

interface AssemblySceneProps {
  trace: TransformerTrace;
  assemblyStep: number; // 0: Base, 1: Embedding, 2: PosEnc, 3: Attention Core, 4: Complete
  isExecuting: boolean;
  activeHead: number;
  cameraResetTrigger: number;
}

function IsometricCameraController({ resetTrigger }: { resetTrigger: number }) {
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0.8, 0);
      controlsRef.current.object.position.set(13, 14, 13);
      controlsRef.current.update();
    }
  }, [resetTrigger]);

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping={true}
      dampingFactor={0.05}
      minDistance={5}
      maxDistance={35}
      maxPolarAngle={Math.PI / 2 - 0.05} // don't go below floor
    />
  );
}

export const AssemblyScene: React.FC<AssemblySceneProps> = ({
  trace,
  assemblyStep,
  isExecuting,
  activeHead,
  cameraResetTrigger
}) => {
  // Assembly flags
  const embeddingAssembled = assemblyStep >= 1;
  const positionalAssembled = assemblyStep >= 2;
  const encoderAssembled = assemblyStep >= 3;

  return (
    <div className="w-full h-full relative bg-[#040711] overflow-hidden select-none">
      <Canvas
        camera={{ position: [13, 14, 13], fov: 38 }}
        gl={{ antialias: true, alpha: false }}
        className="w-full h-full"
      >
        <color attach="background" args={['#040711']} />

        {/* Ambient Dark Tech Starfield */}
        <Stars radius={80} depth={50} count={2000} factor={2} saturation={0.3} fade={true} />

        {/* Industrial Studio Lighting */}
        <ambientLight intensity={0.45} />
        <directionalLight position={[15, 25, 12]} intensity={1.4} castShadow />
        <directionalLight position={[-15, 12, -10]} intensity={0.5} color="#38bdf8" />
        <directionalLight position={[0, -10, 0]} intensity={0.2} color="#00f0ff" />
        <pointLight position={[3.5, 5, 0]} intensity={1.8} color="#a855f7" distance={18} />

        <IsometricCameraController resetTrigger={cameraResetTrigger} />

        {/* 1. Base Cinematic Platform */}
        <BasePlatform
          assemblyProgress={assemblyStep / 4}
          isExecuting={isExecuting}
        />

        {/* 2. Input Embedding Module */}
        <EmbeddingModule
          tokens={trace.encoder.source_tokens}
          embeddings={trace.encoder.embeddings}
          assembled={embeddingAssembled}
          assemblyProgress={assemblyStep >= 1 ? 1 : 0}
          isExecuting={isExecuting}
        />

        {/* 3. Positional Encoding Resonator Module */}
        <PositionalModule
          assembled={positionalAssembled}
          assemblyProgress={assemblyStep >= 2 ? 1 : 0}
          isExecuting={isExecuting}
        />

        {/* 4. Complete Encoder Block (Attention + Add&Norm + FFN) */}
        <EncoderBlockModule
          layerTrace={trace.encoder.layer_1}
          assembled={encoderAssembled}
          assemblyProgress={assemblyStep >= 3 ? 1 : 0}
          isExecuting={isExecuting}
          activeHead={activeHead}
        />

        {/* 5. Inter-Module Signal Bus (Photonic pulses) */}
        <SignalBus isExecuting={isExecuting} />
      </Canvas>
    </div>
  );
};
