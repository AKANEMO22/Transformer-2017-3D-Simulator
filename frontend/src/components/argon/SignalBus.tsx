import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface SignalBusProps {
  isExecuting: boolean;
}

export const SignalBus: React.FC<SignalBusProps> = ({ isExecuting }) => {
  const pulseGroupRef = useRef<THREE.Group>(null);

  // Animate traveling signal pulses along the X-axis from Input (x = -4.5) to Encoder (x = 3.5)
  useFrame((state) => {
    if (!pulseGroupRef.current) return;
    const t = state.clock.getElapsedTime();
    pulseGroupRef.current.children.forEach((child, idx) => {
      if (!isExecuting) {
        child.visible = false;
        return;
      }
      child.visible = true;
      const speed = 2.4;
      const offset = idx * 0.35;
      // Cycles from -4.5 to +3.5
      const cycleProgress = ((t * speed + offset) % 8.0) - 4.5;
      child.position.x = cycleProgress;
    });
  });

  return (
    <group ref={pulseGroupRef} position={[0, -0.25, 0]}>
      {[-2.0, -1.0, 0, 1.0, 2.0].map((z, idx) => (
        <group key={`pulse-${idx}`} position={[-4.5, 0, z]}>
          <mesh>
            <sphereGeometry args={[0.07, 12, 12]} />
            <meshBasicMaterial
              color={idx % 2 === 0 ? '#00f0ff' : '#f43f5e'}
            />
          </mesh>
          <pointLight
            color={idx % 2 === 0 ? '#00f0ff' : '#f43f5e'}
            intensity={1.2}
            distance={1.5}
          />
        </group>
      ))}
    </group>
  );
};
