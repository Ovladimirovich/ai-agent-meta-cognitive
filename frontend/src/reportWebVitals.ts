// Удаляем импорт ReportHandler, так как типы не поддерживаются

const reportWebVitals = (onPerfEntry?: any) => {
  if (onPerfEntry && onPerfEntry instanceof Function) {
    import('web-vitals').then((webVitals: any) => {
      // Используем напрямую методы из объекта webVitals
      if (webVitals.getCLS) webVitals.getCLS(onPerfEntry);
      if (webVitals.getFID) webVitals.getFID(onPerfEntry);
      if (webVitals.getFCP) webVitals.getFCP(onPerfEntry);
      if (webVitals.getLCP) webVitals.getLCP(onPerfEntry);
      if (webVitals.getTTFB) webVitals.getTTFB(onPerfEntry);
    });
  }
};

export default reportWebVitals;
