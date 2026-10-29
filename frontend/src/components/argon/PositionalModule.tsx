import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface PositionalModuleProps {
  assembled: boolean;
  assemblyProgress: number; // 0 to 1
  isExecuting: boolean;
}

export const PositionalModule: React.FC<PositionalModuleProps> = ({
  assembled,
  assemblyProgress,
  isExecuting
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const coilRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      const targetY = assembled ? 0.9 : 0.9 + (1 - Math.min(1, Math.max(0, assemblyProgress))) * 4.0;
      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, targetY, 0.1);
    }
    if (coilRef.current && isExecuting) {
      coilRef.current.rotation.y = state.clock.getElapsedTime() * 1.5;
    }
  });

  return (
    <group ref={groupRef} position={[-4.5, assembled ? 0.9 : 4.9, 0]}>
      {/* Upper Mounting Gantry Frame */}
      <mesh receiveShadow position={[1.8, 0.1, 0]}>
        <boxGeometry args={[1.5, 0.15, 6.8]} />
        <meshStandardMaterial color="#1e293b" metalness={0.9} roughness={0.2} />
      </mesh>

      <Text
        position={[1.8, 0.22, 0]}
        rotation={[-Math.PI / 2, 0, Math.PI / 2]}
        fontSize={0.16}
        color="#fbbf24"
        anchorX="center"
        anchorY="middle"
      >
        MOD-02: POS-ENCODING HARMONIC RESONATOR (SIN/COS)
      </Text>

      {/* Dual Wave Resonator Tubes */}
      <group position={[1.8, 0.6, 0]}>
        {/* Sin Harmonic Tube (Amber) */}
        <mesh position={[-0.3, 0, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.2, 0.2, 5.8, 16]} />
          <meshPhysicalMaterial
            color="#f59e0b"
            emissive="#b45309"
            emissiveIntensity={isExecuting ? 0.8 : 0.2}
            roughness={0.1}
            transmission={0.6}
            thickness={0.5}
          />
        </mesh>

        {/* Cos Harmonic Tube (Cyan/Gold) */}
        <mesh position={[0.3, 0, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.2, 0.2, 5.8, 16]} />
          <meshPhysicalMaterial
            color="#38bdf8"
            emissive="#0284c7"
            emissiveIntensity={isExecuting ? 0.8 : 0.2}
            roughness={0.1}
            transmission={0.6}
            thickness={0.5}
          />
        </mesh>

        {/* Rotating Harmonic Spiral Rings */}
        <group ref={coilRef}>
          {[-2.0, -1.0, 0, 1.0, 2.0].map((z, idx) => (
            <mesh key={`ring-${idx}`} position={[0, 0, z]}>
              <torusGeometry args={[0.65, 0.03, 8, 24]} />
              <meshBasicMaterial
                color={idx % 2 === 0 ? '#fbbf24' : '#38bdf8'}
                transparent={true}
                opacity={0.8}
              />
            </mesh>
          ))}
        </group>
      </group>
    </group>
  );
};
