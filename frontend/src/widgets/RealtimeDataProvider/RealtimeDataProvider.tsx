import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useWebSocket } from '../AdvancedAnalyticsDashboard/WebSocketProvider';

interface RealtimeDataContextType {
  cognitiveData: any;
  memoryData: any;
  reflectionData: any;
  isLoading: boolean;
}

const RealtimeDataContext = createContext<RealtimeDataContextType | undefined>(undefined as any);

interface RealtimeDataProviderProps {
  children: ReactNode;
}

export const RealtimeDataProvider: React.FC<RealtimeDataProviderProps> = ({ children }) => {
  const { ws, isConnected } = useWebSocket();
  const [cognitiveData, setCognitiveData] = useState<any>(null);
  const [memoryData, setMemoryData] = useState<any>(null);
  const [reflectionData, setReflectionData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

 useEffect(() => {
    if (!isConnected) return;

    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'cognitive_update':
            setCognitiveData(data.payload);
            break;
          case 'memory_update':
            setMemoryData(data.payload);
            break;
          case 'reflection_update':
            setReflectionData(data.payload);
            break;
          default:
            console.warn('Unknown message type:', data.type);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws!.addEventListener('message', handleMessage);

    // Имитация начальных данных
    setTimeout(() => {
      setCognitiveData({
        cognitiveLoad: 0.65,
        confidenceLevel: 0.82,
        processingSpeed: 0.78,
        memoryUtilization: 0.58
      });
      setMemoryData({
        memoryLayers: {
          husk: { usage: 0.45, capacity: 1000 },
          soil: { usage: 0.72, capacity: 5000 },
          roots: { usage: 0.38, capacity: 10000 }
        }
      });
      setReflectionData({
        recentReflections: [
          { id: 1, type: 'insight', timestamp: new Date(), confidence: 0.85 }
        ]
      });
      setIsLoading(false);
    }, 500);

    return () => {
      ws!.removeEventListener('message', handleMessage);
    };
  }, [isConnected, ws]);

  return (
    <RealtimeDataContext.Provider value={{ 
      cognitiveData, 
      memoryData, 
      reflectionData, 
      isLoading 
    }}>
      {children}
    </RealtimeDataContext.Provider>
  );
};

export const useRealtimeData = () => {
  const context = useContext(RealtimeDataContext);
  if (context === undefined) {
    throw new Error('useRealtimeData must be used within a RealtimeDataProvider');
  }
  return context;
};