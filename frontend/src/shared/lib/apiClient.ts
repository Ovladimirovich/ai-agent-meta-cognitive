/**
 * Базовая реализация API клиента для фронтенд-приложения
 * Предоставляет общие методы для взаимодействия с backend API
 */

interface ApiConfig {
  baseUrl: string;
  timeout?: number;
  headers?: Record<string, string>;
}

class ApiClient {
  private config: ApiConfig;

  constructor(config: ApiConfig) {
    this.config = {
      baseUrl: config.baseUrl,
      timeout: config.timeout || 10000,
      headers: {
        'Content-Type': 'application/json',
        ...config.headers
      }
    };
  }

  /**
   * Общий метод для выполнения HTTP-запросов
   */
  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;

    const config: RequestInit = {
      ...options,
      headers: {
        ...this.config.headers,
        ...options.headers
      }
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(url, {
        ...config,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data as T;
    } catch (error) {
      clearTimeout(timeoutId);

      if ((error as Error).name === 'AbortError') {
        throw new Error('Request timeout');
      }

      throw error;
    }
  }

  /**
   * GET-запрос
   */
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<T> {
    const queryString = params ? `?${new URLSearchParams(params).toString()}` : '';
    return this.request<T>(`${endpoint}${queryString}`, { method: 'GET' });
  }

  /**
   * POST-запрос
   */
  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined
    });
  }

  /**
   * PUT-запрос
   */
  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined
    });
  }

  /**
   * DELETE-запрос
   */
  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  /**
   * Метод для получения статуса здоровья системы
   */
  async getHealthStatus<T>(): Promise<T> {
    return this.get<T>('/health');
  }

  /**
   * Метод для получения метрик обучения
   */
  async getLearningMetrics<T>(queryString?: string): Promise<T> {
    const endpoint = queryString ? `/learning/metrics?${queryString}` : '/learning/metrics';
    return this.get<T>(endpoint);
  }

  /**
   * Универсальный метод для выполнения запросов с различными целями
   */
  async query<T>(query: string, options?: Record<string, any>): Promise<T> {
    return this.post<T>('/query', { query, ...options });
  }

  /**
   * Метод для обработки запросов агента
   */
  async processRequest<T>(data: any): Promise<T> {
    return this.post<T>('/process', data);
  }

  /**
   * Метод для получения системной информации
   */
  async getSystemInfo<T>(): Promise<T> {
    return this.get<T>('/system/info');
  }
}

// Экземпляр API клиента по умолчанию
const apiClient = new ApiClient({
  baseUrl: (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '') ? '/api' : (process.env.VITE_API_BASE_URL || 'http://localhost:8000/api')
});

export { ApiClient, apiClient };
export type { ApiConfig };
