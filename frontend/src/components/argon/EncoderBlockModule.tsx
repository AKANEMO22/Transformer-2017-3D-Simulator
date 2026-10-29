import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, QuadraticBezierLine } from '@react-three/drei';
import * as THREE from 'three';
import type { LayerAttentionTrace } from '../../types/trace';

interface EncoderBlockModuleProps {
  layerTrace: LayerAttentionTrace;
  assembled: boolean;
  assemblyProgress: number; // 0 to 1
  isExecuting: boolean;
  activeHead: number; // 0, 1, or -1 (all)
}

export const EncoderBlockModule: React.FC<EncoderBlockModuleProps> = ({
  layerTrace,
  assembled,
  assemblyProgress,
  isExecuting,
  activeHead
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const head0CoreRef = useRef<THREE.Group>(null);
  const head1CoreRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      const targetY = assembled ? 0 : (1 - Math.min(1, Math.max(0, assemblyProgress))) * 5.5;
      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, targetY, 0.1);
    }
    const t = state.clock.getElapsedTime();
    if (head0CoreRef.current && isExecuting) {
      head0CoreRef.current.rotation.y = t * 1.2;
    }
    if (head1CoreRef.current && isExecuting) {
      head1CoreRef.current.rotation.y = -t * 1.2;
    }
  });

  const weights = layerTrace?.attention_weights; // [2, 5, 5]

  return (
    <group ref={groupRef} position={[3.5, assembled ? 0 : 5.5, 0]}>
      {/* 1. Main Processor Substrate Base */}
      <mesh receiveShadow position={[0, 0.1, 0]}>
        <boxGeometry args={[7.2, 0.2, 8.2]} />
        <meshStandardMaterial color="#0f172a" metalness={0.8} roughness={0.25} />
      </mesh>

      <lineSegments position={[0, 0.1, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(7.22, 0.22, 8.22)]} />
        <lineBasicMaterial color="#a855f7" transparent={true} opacity={0.8} />
      </lineSegments>

      {/* Module Title Plaque */}
      <Text
        position={[-3.3, 0.24, 0]}
        rotation={[-Math.PI / 2, 0, Math.PI / 2]}
        fontSize={0.2}
        color="#c084fc"
        anchorX="center"
        anchorY="middle"
      >
        MOD-03: COMPLETE ENCODER BLOCK (LAYER 1)
      </Text>

      {/* 2. SUB-ASSEMBLY A: MULTI-HEAD ATTENTION CORE (FRONT BAYS) */}
      <group position={[-0.8, 0.2, 0]}>
        {/* Foundation Carrier */}
        <mesh receiveShadow position={[0, 0.05, 0]}>
          <boxGeometry args={[4.2, 0.1, 7.2]} />
          <meshStandardMaterial color="#1e293b" metalness={0.9} roughness={0.2} />
        </mesh>

        {/* HEAD 0 ROTARY OPTICAL CORE (CYAN - LEFT TOWER) */}
        <group position={[-1.0, 0.6, -1.8]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.7, 0.75, 0.8, 16]} />
            <meshStandardMaterial color="#0284c7" metalness={0.9} roughness={0.2} />
          </mesh>
          <group ref={head0CoreRef}>
            <mesh position={[0, 0.45, 0]}>
              <cylinderGeometry args={[0.55, 0.55, 0.3, 12]} />
              <meshPhysicalMaterial
                color="#00f0ff"
                emissive="#00f0ff"
                emissiveIntensity={isExecuting ? 0.9 : 0.3}
                transmission={0.5}
                roughness={0.1}
              />
            </mesh>
            {/* Collimator Ring */}
            <mesh position={[0, 0.45, 0]}>
              <torusGeometry args={[0.75, 0.04, 8, 24]} />
              <meshBasicMaterial color="#00f0ff" />
            </mesh>
          </group>
          <Text position={[0, 1.1, 0]} fontSize={0.16} color="#38bdf8" anchorX="center">
            HEAD 0 (dk=4)
          </Text>
        </group>

        {/* HEAD 1 ROTARY OPTICAL CORE (MAGENTA - RIGHT TOWER) */}
        <group position={[-1.0, 0.6, 1.8]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.7, 0.75, 0.8, 16]} />
            <meshStandardMaterial color="#be185d" metalness={0.9} roughness={0.2} />
          </mesh>
          <group ref={head1CoreRef}>
            <mesh position={[0, 0.45, 0]}>
              <cylinderGeometry args={[0.55, 0.55, 0.3, 12]} />
              <meshPhysicalMaterial
                color="#f43f5e"
                emissive="#f43f5e"
                emissiveIntensity={isExecuting ? 0.9 : 0.3}
                transmission={0.5}
                roughness={0.1}
              />
            </mesh>
            <mesh position={[0, 0.45, 0]}>
              <torusGeometry args={[0.75, 0.04, 8, 24]} />
              <meshBasicMaterial color="#f43f5e" />
            </mesh>
          </group>
          <Text position={[0, 1.1, 0]} fontSize={0.16} color="#fb7185" anchorX="center">
            HEAD 1 (dk=4)
          </Text>
        </group>

        {/* 5 Input Optical Terminals on Attention Core */}
        {[-2.0, -1.0, 0, 1.0, 2.0].map((z, i) => (
          <group key={`opt-term-${i}`} position={[1.4, 0.25, z]}>
            <mesh castShadow>
              <boxGeometry args={[0.4, 0.3, 0.5]} />
              <meshStandardMaterial color="#334155" metalness={0.9} roughness={0.2} />
            </mesh>
            <mesh position={[0, 0.16, 0]}>
              <cylinderGeometry args={[0.1, 0.1, 0.08, 12]} />
              <meshBasicMaterial color="#00f0ff" />
            </mesh>
            <Text position={[0, 0.35, 0]} fontSize={0.1} color="#94a3b8" anchorX="center">
              {`CH${i}`}
            </Text>
          </group>
        ))}

        {/* LIVE ATTENTION LASER ARCS FIRED BETWEEN CHANNELS */}
        {isExecuting && weights && weights[activeHead === 1 ? 1 : 0]?.map((row, i) => {
          const startZ = (i - 2) * 1.0;
          return row.map((w, j) => {
            if (w < 0.12) return null;
            const endZ = (j - 2) * 1.0;
            const beamColor = activeHead === 1 ? '#f43f5e' : '#00f0ff';
            const midY = 0.5 + Math.abs(i - j) * 0.4 + w * 0.6;

            return (
              <QuadraticBezierLine
                key={`enc-arc-${i}-${j}`}
                start={[1.4, 0.4, startZ]}
                end={[1.4, 0.4, endZ]}
                mid={[0.2, midY, (startZ + endZ) / 2]}
                color={beamColor}
                lineWidth={Math.max(1, w * 4)}
                transparent={true}
                opacity={Math.min(0.85, w * 1.5)}
              />
            );
          });
        })}
      </group>

      {/* 3. SUB-ASSEMBLY B: ADD & LAYERNORM ISOLATION COUPLER */}
      <group position={[1.8, 0.3, 0]}>
        <mesh position={[0, 0.1, 0]}>
          <boxGeometry args={[1.0, 0.2, 6.8]} />
          <meshStandardMaterial color="#1e293b" metalness={0.9} roughness={0.2} />
        </mesh>
        <Text position={[0, 0.3, 0]} rotation={[-Math.PI / 2, 0, Math.PI / 2]} fontSize={0.14} color="#38bdf8" anchorX="center">
          ADD &amp; NORM 1 (LN ε=1e-6)
        </Text>
      </group>

      {/* 4. SUB-ASSEMBLY C: FEED-FORWARD ACCELERATOR (FFN d_ff=16) */}
      <group position={[2.7, 0.3, 0]}>
        {/* Heat dissipation fins (16 fins representing 16-D hidden FFN) */}
        {Array.from({ length: 16 }).map((_, fIdx) => {
          const zOffset = (fIdx - 7.5) * 0.4;
          return (
            <mesh key={`fin-${fIdx}`} position={[0, 0.4, zOffset]} castShadow>
              <boxGeometry args={[0.7, 0.6, 0.08]} />
              <meshStandardMaterial
                color="#0f172a"
                emissive="#a855f7"
                emissiveIntensity={isExecuting ? 0.6 : 0.15}
                metalness={0.9}
                roughness={0.2}
              />
            </mesh>
          );
        })}
        <Text position={[0, 0.9, 0]} fontSize={0.16} color="#e9d5ff" anchorX="center">
          FFN (d_ff=16 ReLU)
        </Text>
      </group>

      {/* 5. RESIDUAL BYPASS CONDUIT (Runs along exterior frame) */}
      <mesh position={[0, 0.6, -3.8]}>
        <boxGeometry args={[6.8, 0.1, 0.15]} />
        <meshStandardMaterial color="#334155" metalness={0.9} roughness={0.2} />
      </mesh>
      <mesh position={[0, 0.68, -3.8]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.02, 0.02, 6.8, 8]} />
        <meshBasicMaterial color="#00f0ff" transparent={true} opacity={isExecuting ? 0.9 : 0.4} />
      </mesh>
      <Text position={[0, 0.85, -3.8]} fontSize={0.12} color="#38bdf8" anchorX="center">
        RESIDUAL BYPASS: x + Sublayer(x)
      </Text>
    </group>
  );
};
