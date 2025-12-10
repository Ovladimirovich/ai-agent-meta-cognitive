import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface MemoryLayer3D {
  // Определение интерфейса для 3D слоя памяти
  nodes: Array<{
    id: string;
    position: [number, number, number];
    value: number;
    connections: string[];
  }>;
  connections: Array<{
    from: string;
    to: string;
    strength: number;
  }>;
}

interface Memory3DVisualizerProps {
  memoryData?: {
    layers: {
      husk: MemoryLayer3D;
      soil: MemoryLayer3D;
      roots: MemoryLayer3D;
    };
  };
  activeLayer?: 'husk' | 'soil' | 'roots';
  className?: string;
}

const Memory3DVisualizer: React.FC<Memory3DVisualizerProps> = ({
  memoryData,
  activeLayer = 'soil',
  className = ''
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<any | null>(null);
  const animationRef = useRef<number>(0);

  // Определение цветов для каждого слоя
  const layerColors = {
    husk: new THREE.Color(0x4F46E5), // indigo
    soil: new THREE.Color(0x10B981), // emerald
    roots: new THREE.Color(0xF59E0B)  // amber
  };

  useEffect(() => {
    if (!containerRef.current) return;

    // Инициализация Three.js сцены
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc); // slate-50
    sceneRef.current = scene;

    // Камера
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 10;
    cameraRef.current = camera;

    // Рендерер
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Управление
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Освещение
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    scene.add(directionalLight);

    // Анимационный цикл
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Очистка при размонтировании
    return () => {
      cancelAnimationFrame(animationRef.current);
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      if (containerRef.current && rendererRef.current?.domElement) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
    };
  }, []);

  // Обновление сцены при изменении данных
  useEffect(() => {
    if (!sceneRef.current || !memoryData) return;

    // Очистка предыдущей сцены
    while (sceneRef.current.children.length > 0) {
      sceneRef.current.remove(sceneRef.current.children[0]);
    }

    // Освещение
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    sceneRef.current.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    sceneRef.current.add(directionalLight);

    // Получение данных активного слоя
    const layerData = memoryData.layers[activeLayer];
    if (!layerData) return;

    // Создание сфер для узлов
    layerData.nodes.forEach(node => {
      const geometry = new THREE.SphereGeometry(node.value * 0.5, 32, 32);
      const material = new THREE.MeshPhongMaterial({
        color: layerColors[activeLayer],
        transparent: true,
        opacity: 0.8
      });
      const sphere = new THREE.Mesh(geometry, material);

      sphere.position.set(...node.position);
      sphere.userData = { id: node.id, value: node.value };
      sceneRef.current!.add(sphere);
    });

    // Создание линий для соединений
    const lineMaterial = new THREE.LineBasicMaterial({
      color: layerColors[activeLayer],
      transparent: true,
      opacity: 0.6
    });

    layerData.connections.forEach(connection => {
      const fromNode = layerData.nodes.find(n => n.id === connection.from);
      const toNode = layerData.nodes.find(n => n.id === connection.to);

      if (fromNode && toNode) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...fromNode.position),
          new THREE.Vector3(...toNode.position)
        ]);

        const line = new THREE.Line(geometry, lineMaterial);
        sceneRef.current!.add(line);
      }
    });

    // Добавление сетки для ориентации
    const gridHelper = new THREE.GridHelper(20, 20, 0xcccccc, 0xcccccc);
    gridHelper.position.y = -5;
    sceneRef.current.add(gridHelper);
  }, [memoryData, activeLayer]);

  // Обработка изменения размера окна
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && cameraRef.current && rendererRef.current) {
        cameraRef.current.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className={`w-full h-full ${className}`}>
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
};

export default Memory3DVisualizer;
