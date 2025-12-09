import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Text, Html, Float } from '@react-three/drei';
import { useQuery } from '@tanstack/react-query';
// Удаляем неиспользуемый импорт
import * as THREE from 'three';

// Определение типов
interface MemoryNode {
  id: string;
  label: string;
  group: string;
  importance?: number;
  color?: string;
  nodeType?: string;
  x?: number;
  y?: number;
  z?: number;
}

interface MemoryLink {
  source: string;
  target: string;
  value?: number;
}

interface MemoryData {
  nodes: MemoryNode[];
  links: MemoryLink[];
}

interface MemoryVisualizerProps {
  className?: string;
}

// Компонент узла для 3D визуализации
const MemoryNode3D: React.FC<{
  node: MemoryNode;
  isSelected: boolean;
  isHighlighted: boolean;
  onClick: (node: MemoryNode) => void;
  onHover: (node: MemoryNode | null) => void;
}> = ({ node, isSelected, isHighlighted, onClick, onHover }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Анимация пульсации для выделенных узлов
  useFrame(() => {
    if (meshRef.current && (isHighlighted || isSelected)) {
      const scale = 1 + Math.sin(Date.now() * 0.005) * 0.1;
      meshRef.current.scale.setScalar(scale);
    } else if (meshRef.current) {
      meshRef.current.scale.setScalar(1);
    }
  });

  // Определение цвета узла на основе типа или важности
  const getColor = () => {
    if (node.color) return node.color;

    // Цвета для разных групп
    const groupColors: Record<string, string> = {
      context: '#6366f1',      // indigo
      knowledge: '#8b5cf6',    // violet
      experience: '#ec4899',   // pink
      default: '#64748b'       // slate
    };

    return groupColors[node.group] || groupColors.default;
  };

  // Размер узла на основе важности
  const size = node.importance ? Math.max(0.2, node.importance * 0.5) : 0.3;

  return (
    <group
      position={[node.x || 0, node.y || 0, node.z || 0]}
      onPointerEnter={(e) => {
        e.stopPropagation();
        setHovered(true);
        onHover(node);
      }}
      onPointerLeave={(e) => {
        e.stopPropagation();
        setHovered(false);
        onHover(null);
      }}
      onClick={(e) => {
        e.stopPropagation();
        onClick(node);
      }}
    >
      <Float
        speed={isSelected ? 1 : 2}
        rotationIntensity={isSelected ? 0.5 : 0.3}
        floatIntensity={isSelected ? 1 : 0.5}
      >
        <Sphere
          ref={meshRef}
          args={[size, 16, 16]}
        >
          <meshStandardMaterial
            color={isSelected ? '#3b82f6' : isHighlighted ? '#f59e0b' : getColor()}
            emissive={hovered ? '#ffffff' : '#0000'}
            emissiveIntensity={hovered ? 0.2 : 0}
            opacity={0.9}
            transparent
          />
        </Sphere>
      </Float>

      {/* Подпись узла при наведении */}
      {hovered && (
        <Html
          position={[0, size + 0.2, 0]}
          center
          distanceFactor={5}
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            pointerEvents: 'none'
          }}
        >
          {node.label}
        </Html>
      )}
    </group>
  );
};

// Компонент связи для 3D визуализации
const MemoryLink3D: React.FC<{
  link: MemoryLink;
  sourceNode: MemoryNode | undefined;
  targetNode: MemoryNode | undefined;
  isHighlighted: boolean;
}> = ({ link, sourceNode, targetNode, isHighlighted }) => {
  if (!sourceNode || !targetNode) return null;

  const sourcePos: [number, number, number] = [
    sourceNode.x || 0,
    sourceNode.y || 0,
    sourceNode.z || 0
  ];

  const targetPos: [number, number, number] = [
    targetNode.x || 0,
    targetNode.y || 0,
    targetNode.z || 0
  ];

  // Рассчитываем вектор направления
  const direction = new THREE.Vector3(...targetPos).sub(new THREE.Vector3(...sourcePos));
  const length = direction.length();
  const midpoint = new THREE.Vector3(...sourcePos).add(direction.clone().multiplyScalar(0.5));

  return (
    <Line
      points={[sourcePos, targetPos]}
      color={isHighlighted ? '#3b82f6' : '#94a3b8'} // blue-500 или gray-400
      lineWidth={isHighlighted ? 2 : 1}
      transparent
      opacity={0.7}
    />
  );
};

// Основная сцена 3D визуализации
const MemoryScene3D: React.FC<{
  memoryData: MemoryData;
  selectedNode: MemoryNode | null;
  highlightNodes: Set<string>;
  onNodeClick: (node: MemoryNode) => void;
  onNodeHover: (node: MemoryNode | null) => void;
}> = ({ memoryData, selectedNode, highlightNodes, onNodeClick, onNodeHover }) => {
  // Настройка камеры
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return (
    <>
      {/* Освещение */}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Сетка для ориентации */}
      <gridHelper args={[20, 20, '#cccccc', '#cccccc']} position={[0, -5, 0]} />

      {/* Узлы */}
      {memoryData.nodes.map((node) => (
        <MemoryNode3D
          key={node.id}
          node={node}
          isSelected={selectedNode?.id === node.id}
          isHighlighted={highlightNodes.has(node.id)}
          onClick={onNodeClick}
          onHover={onNodeHover}
        />
      ))}

      {/* Связи */}
      {memoryData.links.map((link, index) => {
        const sourceNode = memoryData.nodes.find(n => n.id === link.source);
        const targetNode = memoryData.nodes.find(n => n.id === link.target);

        return (
          <MemoryLink3D
            key={`${link.source}-${link.target}`}
            link={link}
            sourceNode={sourceNode}
            targetNode={targetNode}
            isHighlighted={
              (selectedNode &&
                (selectedNode.id === link.source || selectedNode.id === link.target)) ||
              highlightNodes.has(link.source) ||
              highlightNodes.has(link.target)
            }
          />
        );
      })}

      {/* Управление орбитой */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
      />
    </>
  );
};

const MemoryVisualizer: React.FC<MemoryVisualizerProps> = ({ className = '' }) => {
  const [selectedNode, setSelectedNode] = useState<MemoryNode | null>(null);
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());
  const [highlightLinks, setHighlightLinks] = useState<Set<string>>(new Set());

  // Получение данных о памяти с использованием React Query
  const { data: memoryData, isLoading, isError, refetch, isRefetching } = useQuery<MemoryData>({
    queryKey: ['memoryData'],
    queryFn: async () => {
      // В реальной реализации нужно будет использовать apiClient.getMemoryState()
      // Заглушка для получения данных о памяти, так как метода getMemoryState в apiClient нет
      return {
        nodes: [
          {
            id: 'node1',
            label: 'Контекст 1',
            group: 'context',
            importance: 0.8,
            color: '#6366f1',
            x: -2,
            y: 1,
            z: 0
          },
          {
            id: 'node2',
            label: 'Знание 1',
            group: 'knowledge',
            importance: 0.6,
            color: '#8b5cf6',
            x: 2,
            y: -1,
            z: 1
          },
          {
            id: 'node3',
            label: 'Опыт 1',
            group: 'experience',
            importance: 0.9,
            color: '#ec489',
            x: 0,
            y: 2,
            z: -1
          },
          {
            id: 'node4',
            label: 'Знание 2',
            group: 'knowledge',
            importance: 0.4,
            color: '#8b5cf6',
            x: -3,
            y: -2,
            z: 2
          },
          {
            id: 'node5',
            label: 'Контекст 2',
            group: 'context',
            importance: 0.7,
            color: '#6366f1',
            x: 3,
            y: 0,
            z: -2
          }
        ],
        links: [
          { source: 'node1', target: 'node2', value: 0.5 },
          { source: 'node2', target: 'node3', value: 0.7 },
          { source: 'node1', target: 'node4', value: 0.3 },
          { source: 'node3', target: 'node5', value: 0.6 },
          { source: 'node4', target: 'node5', value: 0.4 }
        ]
      };
    },
    refetchInterval: 10000, // Обновление каждые 10 секунд
    staleTime: 5000, // Данные считаются актуальными 5 секунд
    retry: 3, // Повторять запрос при ошибках до 3 раз
    retryDelay: 1000, // Задержка между повторами
  });

  // Обработчик выбора узла
  const handleNodeClick = useCallback((node: MemoryNode) => {
    setSelectedNode(node);
    // Подсвечиваем связанные узлы и связи
    if (memoryData) {
      const connectedNodes = new Set<string>();
      const connectedLinks = new Set<string>();

      memoryData.links.forEach((link, index) => {
        if (link.source === node.id || link.target === node.id) {
          connectedNodes.add(link.source.toString());
          connectedNodes.add(link.target.toString());
          connectedLinks.add(`${index}`);
        }
      });

      setHighlightNodes(connectedNodes);
      setHighlightLinks(connectedLinks);
    }
  }, [memoryData]);

  // Обработчик наведения на узел
  const handleNodeHover = useCallback((node: MemoryNode | null) => {
    if (node && memoryData) {
      const connectedNodes = new Set<string>();

      memoryData.links.forEach((link) => {
        if (link.source === node.id || link.target === node.id) {
          connectedNodes.add(link.source.toString());
          connectedNodes.add(link.target.toString());
        }
      });

      setHighlightNodes(connectedNodes);
    } else {
      setHighlightNodes(new Set());
    }
  }, [memoryData]);

  // Обновление данных при изменении
  useEffect(() => {
    if (!memoryData) return;

    // Сброс подсветки при обновлении данных
    setHighlightNodes(new Set());
    setHighlightLinks(new Set());
    setSelectedNode(null);
  }, [memoryData]);

  if (isLoading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-gray-600 dark:text-gray-300">Загрузка визуализации памяти...</p>
          </div>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
        <div className="text-center py-8">
          <div className="text-red-500 text-2xl mb-2">⚠️</div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-1">Ошибка загрузки данных</h3>
          <p className="text-gray-600 dark:text-gray-300 mb-4">Не удалось загрузить данные о состоянии памяти</p>
          <button
            onClick={() => refetch()}
            disabled={isRefetching}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {isRefetching ? 'Повтор...' : 'Повторить попытку'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white">Визуализация Памяти</h2>
        <div className="text-sm text-gray-600 dark:text-gray-300 flex items-center space-x-4">
          <span>Узлов: {memoryData?.nodes.length || 0} | Связей: {memoryData?.links.length || 0}</span>
          {isRefetching && <span className="text-xs text-blue-500">Обновление...</span>}
        </div>
      </div>

      {memoryData && memoryData.nodes.length > 0 ? (
        <div className="h-96 md:h-[400px] lg:h-[500px] xl:h-[600px] relative">
          <Canvas
            camera={{ position: [10, 10, 10], fov: 50 }}
            onCreated={({ gl }) => {
              // Адаптивная настройка рендерера для разных экранов
              const handleResize = () => {
                gl.setSize(
                  document.querySelector('.h-96')?.clientWidth || window.innerWidth,
                  document.querySelector('.h-96')?.clientHeight || 400
                );
              };

              window.addEventListener('resize', handleResize);
              handleResize(); // Инициализация размера

              return () => window.removeEventListener('resize', handleResize);
            }}
          >
            <MemoryScene3D
              memoryData={memoryData}
              selectedNode={selectedNode}
              highlightNodes={highlightNodes}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
            />
          </Canvas>
        </div>
      ) : (
        <div className="flex items-center justify-center h-96 md:h-[400px] lg:h-[500px] xl:h-[600px] text-gray-500 dark:text-gray-400">
          <div className="text-center">
            <div className="text-2xl mb-2">🧠</div>
            <p className="text-lg font-medium">Нет данных о памяти</p>
            <p className="text-sm">Система памяти пока пуста</p>
          </div>
        </div>
      )}

      {/* Информация о выбранном узле */}
      {selectedNode && (
        <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border-gray-200 dark:border-gray-600">
          <h3 className="font-medium text-gray-900 dark:text-white mb-2">Детали узла</h3>
          <div className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
            <p><span className="font-medium">ID:</span> {selectedNode.id}</p>
            <p><span className="font-medium">Метка:</span> {selectedNode.label}</p>
            <p><span className="font-medium">Группа:</span> {selectedNode.group}</p>
            <p><span className="font-medium">Тип:</span> {selectedNode.nodeType}</p>
            <p><span className="font-medium">Значимость:</span> {(selectedNode.importance || 0).toFixed(2)}</p>
          </div>
        </div>
      )}

      {/* Легенда */}
      <div className="mt-4 flex flex-wrap gap-2 text-xs text-gray-600 dark:text-gray-400">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-blue-500 mr-1"></div>
          <span>Выбранный узел</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-gray-400 mr-1"></div>
          <span>Обычный узел</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-blue-50 mr-1"></div>
          <span>Подсвеченная связь</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-amber-500 mr-1"></div>
          <span>Узел при наведении</span>
        </div>
      </div>
    </div>
  );
};

export default MemoryVisualizer;
