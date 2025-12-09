/**
 * Общие утилиты для фронтенд-приложения
 * Содержит часто используемые функции и вспомогательные методы
 */

/**
 * Форматирует дату в удобочитаемый формат
 */
export const formatDate = (date: Date | string | number): string => {
  if (typeof date === 'string') {
    date = new Date(date);
  } else if (typeof date === 'number') {
    date = new Date(date);
  }

  return new Intl.DateTimeFormat('ru-RU', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  }).format(date);
};

/**
 * Генерирует уникальный идентификатор
 */
export const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
};

/**
 * Проверяет, является ли значение пустым
 */
export const isEmpty = (value: any): boolean => {
  if (value === null || value === undefined) {
    return true;
  }

  if (typeof value === 'string' && value.trim() === '') {
    return true;
  }

  if (Array.isArray(value) && value.length === 0) {
    return true;
  }

  if (typeof value === 'object' && Object.keys(value).length === 0) {
    return true;
  }

  return false;
};

/**
 * Задержка выполнения (асинхронная)
 */
export const delay = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

/**
 * Обертка для безопасного доступа к вложенным свойствам объекта
 */
export const safeGet = <T>(obj: any, path: string, defaultValue?: T): T | undefined => {
  try {
    const keys = path.split('.');
    let result = obj;

    for (const key of keys) {
      if (result === null || result === undefined) {
        return defaultValue;
      }
      result = result[key];
    }

    return result !== undefined ? result as T : defaultValue;
  } catch {
    return defaultValue;
  }
};

/**
 * Функция для глубокого копирования объекта
 */
export const deepClone = <T>(obj: T): T => {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as any;
  }

  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as any;
  }

  if (typeof obj === 'object') {
    const clonedObj: any = {};
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = deepClone(obj[key]);
      }
    }
    return clonedObj;
  }

  return obj as T;
};

/**
 * Функция для слияния объектов (объединение)
 */
export const mergeObjects = <T>(target: T, source: Partial<T>): T => {
  const result = { ...target };

  for (const key in source) {
    if (source.hasOwnProperty(key)) {
      if (
        typeof source[key] === 'object' &&
        source[key] !== null &&
        !Array.isArray(source[key]) &&
        typeof result[key] === 'object' &&
        result[key] !== null &&
        !Array.isArray(result[key])
      ) {
        result[key] = mergeObjects(result[key], source[key]);
      } else {
        result[key] = source[key] as any;
      }
    }
  }

  return result;
};

/**
 * Функция для преобразования объекта в URL-параметры
 */
export const objectToUrlParams = (obj: Record<string, any>): string => {
  const params = new URLSearchParams();

  for (const key in obj) {
    if (obj.hasOwnProperty(key) && obj[key] !== undefined && obj[key] !== null) {
      params.append(key, String(obj[key]));
    }
  }

  return params.toString();
};

/**
 * Функция для обработки ошибок с логированием
 */
export const handleAsyncError = async <T>(
  promise: Promise<T>,
  onError?: (error: Error) => void
): Promise<T | null> => {
  try {
    return await promise;
  } catch (error) {
    console.error('Async operation error:', error);
    if (onError) {
      onError(error as Error);
    }
    return null;
  }
};

/**
 * Функция для объединения CSS-классов
 */
export const cn = (...classes: (string | boolean | undefined)[]): string => {
  return classes.filter(Boolean).join(' ');
};
