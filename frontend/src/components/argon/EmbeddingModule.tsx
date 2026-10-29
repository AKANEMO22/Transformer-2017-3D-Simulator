import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface EmbeddingModuleProps {
  tokens: string[];
  embeddings: number[][]; // [5, 8]
  assembled: boolean;
  assemblyProgress: number; // 0 to 1 for this module
  isExecuting: boolean;
}

export const EmbeddingModule: React.FC<EmbeddingModuleProps> = ({
  tokens,
  embeddings,
  assembled,
  assemblyProgress,
  isExecuting
}) => {
  const groupRef = useRef<THREE.Group>(null);

  // Smooth lerp for assembly descent: 0 -> elevated at y=4.5, 1 -> docked at y=0
  useFrame(() => {
    if (groupRef.current) {
      const targetY = assembled ? 0 : (1 - Math.min(1, Math.max(0, assemblyProgress))) * 4.5;
      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, targetY, 0.1);
    }
  });

  const spacing = 1.35;
  const startZ = -((tokens.length - 1) * spacing) / 2;

  return (
    <group ref={groupRef} position={[-4.5, assembled ? 0 : 4.5, 0]}>
      {/* 1. Module Main Board / Substrate */}
      <mesh receiveShadow position={[0, 0.1, 0]}>
        <boxGeometry args={[5.6, 0.2, 7.2]} />
        <meshStandardMaterial
          color="#0f172a"
          metalness={0.7}
          roughness={0.3}
        />
      </mesh>

      {/* Board Edge Chamfer Trim */}
      <lineSegments position={[0, 0.1, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(5.62, 0.22, 7.22)]} />
        <lineBasicMaterial color="#00f0ff" transparent={true} opacity={0.7} />
      </lineSegments>

      {/* Module Title Plaque */}
      <mesh position={[-2.4, 0.22, 0]} rotation={[-Math.PI / 2, 0, Math.PI / 2]}>
        <planeGeometry args={[6.5, 0.35]} />
        <meshBasicMaterial color="#0284c7" transparent={true} opacity={0.3} />
      </mesh>
      <Text
        position={[-2.35, 0.23, 0]}
        rotation={[-Math.PI / 2, 0, Math.PI / 2]}
        fontSize={0.2}
        color="#38bdf8"
        anchorX="center"
        anchorY="middle"
      >
        MOD-01: INPUT TOKEN EMBEDDING MATRIX [5x8]
      </Text>

      {/* 2. Vertical Alignment Rails (Guide Lasers when assembling) */}
      {!assembled && (
        <group>
          {[-2.6, 2.6].map((x, xi) =>
            [-3.4, 3.4].map((z, zi) => (
              <mesh key={`guide-${xi}-${zi}`} position={[x, -2.0, z]}>
                <cylinderGeometry args={[0.015, 0.015, 4.5, 6]} />
                <meshBasicMaterial color="#00f0ff" transparent={true} opacity={0.4} />
              </mesh>
            ))
          )}
        </group>
      )}

      {/* 3. Token Micro-Die Sockets */}
      {tokens.map((token, idx) => {
        const zPos = startZ + idx * spacing;
        const vec = embeddings[idx] || [0, 0, 0, 0, 0, 0, 0, 0];

        return (
          <group key={`chip-${idx}`} position={[0, 0.2, zPos]}>
            {/* Socket Carrier */}
            <mesh receiveShadow position={[-0.8, 0.05, 0]}>
              <boxGeometry args={[2.6, 0.12, 1.1]} />
              <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
            </mesh>

            {/* Token Ceramic Package */}
            <mesh position={[-1.4, 0.16, 0]} castShadow>
              <boxGeometry args={[1.2, 0.14, 0.9]} />
              <meshPhysicalMaterial
                color="#0f172a"
                emissive="#0284c7"
                emissiveIntensity={isExecuting ? 0.4 : 0.15}
                roughness={0.2}
                metalness={0.9}
              />
            </mesh>

            {/* Token Text Engraving */}
            <Text
              position={[-1.4, 0.24, 0]}
              rotation={[-Math.PI / 2, 0, 0]}
              fontSize={0.22}
              color="#ffffff"
              anchorX="center"
              anchorY="middle"
            >
              {token}
            </Text>
            <Text
              position={[-1.4, 0.24, 0.32]}
              rotation={[-Math.PI / 2, 0, 0]}
              fontSize={0.11}
              color="#38bdf8"
              anchorX="center"
              anchorY="middle"
            >
              {`ID:${idx + 3}`}
            </Text>

            {/* 8-Dimensional Vector Pins (d_model = 8) */}
            <group position={[0.2, 0.12, 0]}>
              {vec.map((val, dIdx) => {
                const xOffset = (dIdx - 3.5) * 0.22;
                const pinHeight = Math.max(0.12, Math.abs(val) * 0.6 + 0.15);
                const pinColor = val >= 0 ? '#00f0ff' : '#f43f5e';

                return (
                  <group key={`pin-${dIdx}`} position={[xOffset, 0, 0]}>
                    <mesh position={[0, pinHeight / 2, 0]} castShadow>
                      <cylinderGeometry args={[0.04, 0.04, pinHeight, 8]} />
                      <meshStandardMaterial
                        color={pinColor}
                        emissive={pinColor}
                        emissiveIntensity={isExecuting ? 0.7 : 0.3}
                        metalness={0.5}
                        roughness={0.2}
                      />
                    </mesh>
                  </group>
                );
              })}
            </group>
          </group>
        );
      })}
    </group>
  );
};
