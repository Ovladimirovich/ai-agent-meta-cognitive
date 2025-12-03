import React from 'react';

interface ZustandProviderProps {
  children: React.ReactNode;
}

// Провайдер Zustand для всего приложения
export const ZustandProvider: React.FC<ZustandProviderProps> = ({ children }) => {
  return (
    <>
      {children}
    </>
  );
};