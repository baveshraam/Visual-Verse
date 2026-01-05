import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Points, PointMaterial, Float, Icosahedron, Octahedron, Torus } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette, ChromaticAberration } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import * as THREE from 'three';
import './Background3D.css';

// Generate random points in a sphere
function generateSpherePoints(count, radius) {
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = radius * Math.cbrt(Math.random()); // Cube root for uniform distribution

        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);
    }

    return positions;
}

// Animated particle field
function ParticleField({ count = 5000, radius = 15 }) {
    const ref = useRef();
    const colorRef = useRef(0);

    const positions = useMemo(() => generateSpherePoints(count, radius), [count, radius]);

    // Create color array for gradient effect
    const colors = useMemo(() => {
        const colorArray = new Float32Array(count * 3);
        const color1 = new THREE.Color('#00f5ff');
        const color2 = new THREE.Color('#b829ff');
        const color3 = new THREE.Color('#ff2d7a');

        for (let i = 0; i < count; i++) {
            const t = Math.random();
            const color = t < 0.33
                ? color1.clone().lerp(color2, t * 3)
                : t < 0.66
                    ? color2.clone().lerp(color3, (t - 0.33) * 3)
                    : color3.clone().lerp(color1, (t - 0.66) * 3);

            colorArray[i * 3] = color.r;
            colorArray[i * 3 + 1] = color.g;
            colorArray[i * 3 + 2] = color.b;
        }

        return colorArray;
    }, [count]);

    useFrame((state, delta) => {
        if (ref.current) {
            ref.current.rotation.x += delta * 0.02;
            ref.current.rotation.y += delta * 0.03;

            // Subtle pulsing
            const scale = 1 + Math.sin(state.clock.elapsedTime * 0.5) * 0.05;
            ref.current.scale.set(scale, scale, scale);
        }
    });

    return (
        <Points ref={ref} positions={positions} colors={colors} stride={3} frustumCulled={false}>
            <PointMaterial
                transparent
                vertexColors
                size={0.02}
                sizeAttenuation={true}
                depthWrite={false}
                opacity={0.8}
                blending={THREE.AdditiveBlending}
            />
        </Points>
    );
}

// Inner particle cloud (denser, smaller)
function InnerCloud({ count = 2000 }) {
    const ref = useRef();
    const positions = useMemo(() => generateSpherePoints(count, 5), [count]);

    useFrame((state, delta) => {
        if (ref.current) {
            ref.current.rotation.x -= delta * 0.05;
            ref.current.rotation.z += delta * 0.04;
        }
    });

    return (
        <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
            <PointMaterial
                transparent
                color="#00f5ff"
                size={0.015}
                sizeAttenuation={true}
                depthWrite={false}
                opacity={0.6}
                blending={THREE.AdditiveBlending}
            />
        </Points>
    );
}

// Floating geometric shapes
function FloatingGeometry() {
    const icosaRef = useRef();
    const octaRef = useRef();
    const torusRef = useRef();

    useFrame((state) => {
        const t = state.clock.elapsedTime;

        if (icosaRef.current) {
            icosaRef.current.rotation.x = t * 0.2;
            icosaRef.current.rotation.y = t * 0.3;
            icosaRef.current.position.y = Math.sin(t * 0.5) * 0.5;
        }

        if (octaRef.current) {
            octaRef.current.rotation.x = t * 0.3;
            octaRef.current.rotation.z = t * 0.2;
            octaRef.current.position.x = Math.sin(t * 0.4) * 0.5;
        }

        if (torusRef.current) {
            torusRef.current.rotation.x = t * 0.15;
            torusRef.current.rotation.y = t * 0.25;
        }
    });

    const glassMaterial = useMemo(() => (
        <meshPhysicalMaterial
            color="#ffffff"
            metalness={0.1}
            roughness={0.1}
            transmission={0.9}
            thickness={0.5}
            transparent
            opacity={0.3}
        />
    ), []);

    return (
        <>
            <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
                <Icosahedron ref={icosaRef} args={[0.8, 1]} position={[-4, 2, -3]}>
                    <meshPhysicalMaterial
                        color="#00f5ff"
                        metalness={0.2}
                        roughness={0.1}
                        transmission={0.8}
                        thickness={0.5}
                        transparent
                        opacity={0.2}
                        wireframe
                    />
                </Icosahedron>
            </Float>

            <Float speed={1.5} rotationIntensity={0.8} floatIntensity={1.5}>
                <Octahedron ref={octaRef} args={[0.6, 0]} position={[4, -1, -4]}>
                    <meshPhysicalMaterial
                        color="#b829ff"
                        metalness={0.3}
                        roughness={0.2}
                        transmission={0.7}
                        transparent
                        opacity={0.25}
                        wireframe
                    />
                </Octahedron>
            </Float>

            <Float speed={1} rotationIntensity={0.3} floatIntensity={0.8}>
                <Torus ref={torusRef} args={[1.2, 0.1, 16, 100]} position={[0, -3, -5]}>
                    <meshPhysicalMaterial
                        color="#ff2d7a"
                        metalness={0.4}
                        roughness={0.1}
                        transmission={0.6}
                        transparent
                        opacity={0.2}
                        wireframe
                    />
                </Torus>
            </Float>
        </>
    );
}

// Mouse-reactive camera
function CameraController() {
    const { camera } = useThree();
    const mouse = useRef({ x: 0, y: 0 });
    const target = useRef({ x: 0, y: 0 });

    useEffect(() => {
        const handleMouseMove = (e) => {
            mouse.current.x = (e.clientX / window.innerWidth) * 2 - 1;
            mouse.current.y = -(e.clientY / window.innerHeight) * 2 + 1;
        };

        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);

    useFrame(() => {
        // Smooth interpolation
        target.current.x += (mouse.current.x * 0.5 - target.current.x) * 0.05;
        target.current.y += (mouse.current.y * 0.5 - target.current.y) * 0.05;

        camera.position.x = target.current.x;
        camera.position.y = target.current.y;
        camera.lookAt(0, 0, 0);
    });

    return null;
}

// Ambient lighting
function Lighting() {
    return (
        <>
            <ambientLight intensity={0.2} />
            <pointLight position={[10, 10, 10]} color="#00f5ff" intensity={0.5} />
            <pointLight position={[-10, -10, -10]} color="#b829ff" intensity={0.3} />
            <pointLight position={[0, 10, -10]} color="#ff2d7a" intensity={0.2} />
        </>
    );
}

// Post-processing effects
function Effects() {
    return (
        <EffectComposer>
            <Bloom
                intensity={0.8}
                luminanceThreshold={0.2}
                luminanceSmoothing={0.9}
                blendFunction={BlendFunction.SCREEN}
            />
            <Vignette
                offset={0.3}
                darkness={0.7}
                blendFunction={BlendFunction.NORMAL}
            />
            <ChromaticAberration
                offset={[0.0005, 0.0005]}
                blendFunction={BlendFunction.NORMAL}
            />
        </EffectComposer>
    );
}

// Main Background3D component
export default function Background3D() {
    return (
        <div className="background-3d">
            <Canvas
                camera={{ position: [0, 0, 8], fov: 60 }}
                dpr={[1, 2]}
                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: 'high-performance'
                }}
            >
                <color attach="background" args={['#0a0a0f']} />
                <fog attach="fog" args={['#0a0a0f', 8, 25]} />

                <CameraController />
                <Lighting />

                <ParticleField count={6000} radius={18} />
                <InnerCloud count={2500} />
                <FloatingGeometry />

                <Effects />
            </Canvas>
        </div>
    );
}
