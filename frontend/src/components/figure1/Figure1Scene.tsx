import React, { useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import type { TransformerTrace } from '../../types/trace';

import specData from '../../spec/transformer_2017_spec.json';
import { Figure1Node } from './Figure1Node';
import { Figure1Edge } from './Figure1Edge';
import { ExplodedLayerDetail } from './ExplodedLayerDetail';

interface Figure1SceneProps {
  trace: TransformerTrace;
  viewMode: 'global' | 'enc_exploded' | 'dec_exploded' | 'cross_focus' | 'causal_focus';
  isExecuting: boolean;
  selectedNodeId: string | null;
  onSelectNode: (id: string | null) => void;
  resetTrigger: number;
}

function SceneCameraController({
  viewMode,
  resetTrigger
}: {
  viewMode: string;
  resetTrigger: number;
}) {
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    if (!controlsRef.current) return;

    if (viewMode === 'global') {
      controlsRef.current.target.set(0, 3.5, 0);
      controlsRef.current.object.position.set(16, 6, 22);
    } else if (viewMode === 'enc_exploded') {
      controlsRef.current.target.set(0, 3.5, 3.5);
      controlsRef.current.object.position.set(0, 3.5, 24);
    } else if (viewMode === 'dec_exploded') {
      controlsRef.current.target.set(0, 3.5, 3.5);
      controlsRef.current.object.position.set(0, 3.5, 24);
    } else if (viewMode === 'cross_focus') {
      controlsRef.current.target.set(0, 4.0, 0);
      controlsRef.current.object.position.set(0, 4.5, 18);
    } else if (viewMode === 'causal_focus') {
      controlsRef.current.target.set(4.2, -1.0, 0);
      controlsRef.current.object.position.set(4.2, -0.5, 14);
    }
    controlsRef.current.update();
  }, [viewMode, resetTrigger]);

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping={true}
      dampingFactor={0.05}
      minDistance={4}
      maxDistance={40}
    />
  );
}

export const Figure1Scene: React.FC<Figure1SceneProps> = ({
  trace,
  viewMode,
  isExecuting,
  selectedNodeId,
  onSelectNode,
  resetTrigger
}) => {
  const nodesMap = React.useMemo(() => {
    const map = new Map<string, any>();
    specData.nodes.forEach((n) => map.set(n.id, n));
    return map;
  }, []);

  const isExploded = viewMode === 'enc_exploded' || viewMode === 'dec_exploded';

  return (
    <div className="w-full h-full relative bg-[#060a14] overflow-hidden select-none">
      <Canvas
        camera={{ position: [16, 6, 22], fov: 38 }}
        gl={{ antialias: true, alpha: false }}
        className="w-full h-full"
      >
        <color attach="background" args={['#060a14']} />

        {/* Clean Scientific Studio Lighting */}
        <ambientLight intensity={0.65} />
        <directionalLight position={[18, 25, 20]} intensity={1.3} castShadow />
        <directionalLight position={[-18, 15, -10]} intensity={0.4} color="#38bdf8" />
        <directionalLight position={[0, -10, 0]} intensity={0.2} color="#00f0ff" />

        <SceneCameraController viewMode={viewMode} resetTrigger={resetTrigger} />

        {/* Ground Reference Grid */}
        <gridHelper args={[32, 32, '#1e293b', '#0f172a']} position={[0, -6.5, 0]} />

        {/* Column Bracket Titles */}
        <group position={[-4.2, 8.2, 0]}>
          <Text fontSize={0.36} color="#38bdf8" anchorX="center">
            ENCODER STACK (Nx=2)
          </Text>
          <Text position={[0, -0.4, 0]} fontSize={0.16} color="#94a3b8" anchorX="center">
            Paper Nx=6 | Self-Attn + Add&amp;Norm + FFN
          </Text>
        </group>

        <group position={[4.2, 14.2, 0]}>
          <Text fontSize={0.36} color="#fb923c" anchorX="center">
            DECODER STACK (Nx=2)
          </Text>
          <Text position={[0, -0.4, 0]} fontSize={0.16} color="#94a3b8" anchorX="center">
            Masked Self-Attn + Cross-Attn + FFN + Linear &amp; Softmax
          </Text>
        </group>

        {/* 1. DYNAMICALLY RENDER ALL EDGES FROM SPEC */}
        {specData.edges.map((edge, idx) => {
          const fromNode = nodesMap.get(edge.from);
          const toNode = nodesMap.get(edge.to);
          if (!fromNode || !toNode) return null;

          return (
            <Figure1Edge
              key={`edge-${idx}`}
              fromPos={fromNode.position as [number, number, number]}
              toPos={toNode.position as [number, number, number]}
              type={edge.type as any}
              isExecuting={isExecuting}
            />
          );
        })}

        {/* 2. DYNAMICALLY RENDER ALL NODES FROM SPEC */}
        {specData.nodes.map((node) => {
          const isSelected = selectedNodeId === node.id;

          return (
            <Figure1Node
              key={node.id}
              id={node.id}
              label={node.label}
              subLabel={(node as any).sub_label}
              type={node.type}
              position={node.position as [number, number, number]}
              size={node.size as [number, number, number]}
              color={node.color}
              isHighlighted={isSelected}
              isExecuting={isExecuting}
              onClick={() => onSelectNode(isSelected ? null : node.id)}
            />
          );
        })}

        {/* 3. EXPLODED SUBLAYER DETAIL OVERLAY (WHEN TOGGLED) */}
        {isExploded && (
          <ExplodedLayerDetail
            layerType={viewMode === 'enc_exploded' ? 'encoder' : 'decoder'}
            layerIndex={0}
            encoderTrace={trace.encoder.layer_1}
            decoderStepTrace={trace.decoder_steps?.[0]}
            onClose={() => onSelectNode(null)}
          />
        )}
      </Canvas>
    </div>
  );
};
