import React from 'react';
import * as THREE from 'three';

interface BasePlatformProps {
  assemblyProgress: number; // 0 to 1
  isExecuting: boolean;
}

export const BasePlatform: React.FC<BasePlatformProps> = ({ isExecuting }) => {
  return (
    <group position={[0, -0.6, 0]}>
      {/* 1. Main Graphite Chassis Plate */}
      <mesh receiveShadow position={[0, -0.2, 0]}>
        <boxGeometry args={[18, 0.4, 12]} />
        <meshStandardMaterial
          color="#0a0e17"
          roughness={0.4}
          metalness={0.8}
        />
      </mesh>

      {/* 2. Beveled Metallic Outer Frame */}
      <mesh receiveShadow position={[0, -0.42, 0]}>
        <boxGeometry args={[18.6, 0.25, 12.6]} />
        <meshStandardMaterial
          color="#06090f"
          roughness={0.2}
          metalness={0.9}
        />
      </mesh>

      {/* 3. Corner Mounting Brackets (4 industrial corners) */}
      {[
        [-9.1, -0.2, -6.1],
        [9.1, -0.2, -6.1],
        [-9.1, -0.2, 6.1],
        [9.1, -0.2, 6.1]
      ].map((pos, idx) => (
        <group key={`bracket-${idx}`} position={pos as [number, number, number]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.5, 0.6, 0.6, 8]} />
            <meshStandardMaterial color="#1e293b" metalness={0.9} roughness={0.2} />
          </mesh>
          {/* Bolt cap */}
          <mesh position={[0, 0.35, 0]}>
            <cylinderGeometry args={[0.25, 0.25, 0.1, 6]} />
            <meshStandardMaterial color="#64748b" metalness={1.0} roughness={0.1} />
          </mesh>
        </group>
      ))}

      {/* 4. Socket 0: Input & Embedding Dock (Left side) */}
      <group position={[-4.5, 0.02, 0]}>
        {/* Recessed Dock Bed */}
        <mesh receiveShadow>
          <boxGeometry args={[6.8, 0.05, 7.8]} />
          <meshStandardMaterial color="#030712" roughness={0.6} metalness={0.7} />
        </mesh>
        {/* Dock Glowing Border */}
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(6.85, 0.06, 7.85)]} />
          <lineBasicMaterial color="#00f0ff" transparent={true} opacity={0.6} />
        </lineSegments>
        {/* Dock Label */}
        <mesh position={[0, 0.03, -3.6]}>
          <planeGeometry args={[3.0, 0.3]} />
          <meshBasicMaterial color="#00f0ff" transparent={true} opacity={0.2} />
        </mesh>
      </group>

      {/* 5. Socket 1: Encoder Core Processor Dock (Right side) */}
      <group position={[3.5, 0.02, 0]}>
        <mesh receiveShadow>
          <boxGeometry args={[7.8, 0.05, 8.8]} />
          <meshStandardMaterial color="#030712" roughness={0.6} metalness={0.7} />
        </mesh>
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(7.85, 0.06, 8.85)]} />
          <lineBasicMaterial color="#a855f7" transparent={true} opacity={0.6} />
        </lineSegments>
      </group>

      {/* 6. Engraved High-Speed Bus Conduits (Connecting Left to Right) */}
      {[-2.0, -1.0, 0, 1.0, 2.0].map((zOffset, i) => (
        <group key={`bus-${i}`} position={[-0.5, 0.02, zOffset]}>
          {/* Conduit Trench */}
          <mesh receiveShadow>
            <boxGeometry args={[3.2, 0.03, 0.12]} />
            <meshStandardMaterial color="#020617" roughness={0.8} />
          </mesh>
          {/* Glowing Optical Wire */}
          <mesh position={[0, 0.02, 0]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.02, 0.02, 3.2, 8]} />
            <meshBasicMaterial
              color={i % 2 === 0 ? '#00f0ff' : '#f43f5e'}
              transparent={true}
              opacity={isExecuting ? 0.9 : 0.5}
            />
          </mesh>
        </group>
      ))}

      {/* 7. Grid Matrix Texture on Floor */}
      <gridHelper args={[24, 24, '#00f0ff', '#1e293b']} position={[0, -0.45, 0]} />
    </group>
  );
};
