import { ExportData, ImportResult, ExportOptions, ImportOptions } from './types/import-export';

const APP_VERSION = '1.0.0';

export class ImportExportService {
  /**
   * Экспорт данных в файл
   * @param data Данные для экспорта
   * @param type Тип данных (используется для имени файла и валидации)
   * @param options Дополнительные настройки экспорта
   */
  static async exportData<T>(
    data: T,
    type: string,
    options: ExportOptions = { format: 'json' }
  ): Promise<boolean> {
    try {
      const exportData: ExportData<T> = {
        version: APP_VERSION,
        timestamp: new Date().toISOString(),
        type,
        data
      };

      const content = options.format === 'json'
        ? JSON.stringify(exportData, null, 2)
        : this.convertToFormat(exportData, options.format);

      const blob = new Blob([content], { type: this.getMimeType(options.format) });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = this.generateFileName(type, options);
      document.body.appendChild(a);
      a.click();

      // Очистка
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 100);

      return true;
    } catch (error) {
      console.error('Export failed:', error);
      return false;
    }
  }

  /**
   * Импорт данных из файла
   * @param file Файл для импорта
   * @param options Настройки импорта
   */
  static async importData<T>(
    file: File,
    options: ImportOptions = {}
  ): Promise<ImportResult<T>> {
    try {
      const content = await file.text();
      let data: ExportData<T>;

      try {
        data = JSON.parse(content) as ExportData<T>;
      } catch (e) {
        return {
          success: false,
          message: 'Неверный формат файла. Ожидается JSON.',
          errors: ['INVALID_JSON']
        };
      }

      // Валидация схемы, если требуется
      if (options.validateSchema) {
        const validation = this.validateImportData(data, options.requiredFields);
        if (!validation.valid) {
          return {
            success: false,
            message: 'Ошибка валидации данных',
            errors: validation.errors
          };
        }
      }

      return {
        success: true,
        message: 'Данные успешно импортированы',
        data: data.data
      };
    } catch (error) {
      console.error('Import failed:', error);
      return {
        success: false,
        message: 'Произошла ошибка при импорте данных',
        errors: ['IMPORT_ERROR']
      };
    }
  }

  private static generateFileName(type: string, options: ExportOptions): string {
    const date = new Date().toISOString().slice(0, 10);
    const ext = options.format || 'json';
    const name = options.customFileName || `export_${type}_${date}`;
    return `${name}.${ext}`;
  }

  private static getMimeType(format: string = 'json'): string {
    const mimeTypes: Record<string, string> = {
      'json': 'application/json',
      'csv': 'text/csv',
      'txt': 'text/plain'
    };
    return mimeTypes[format] || 'application/octet-stream';
  }

  private static convertToFormat(data: any, format: string = 'json'): string {
    switch (format) {
      case 'csv':
        return this.convertToCSV(data);
      case 'txt':
        return this.convertToTXT(data);
      default:
        return JSON.stringify(data, null, 2);
    }
  }

  private static convertToCSV(data: any): string {
    // Простая реализация конвертации в CSV
    if (Array.isArray(data)) {
      if (data.length === 0) return '';

      const headers = Object.keys(data[0]);
      const rows = data.map((item: any) =>
        headers.map(header =>
          `"${String(item[header] || '').replace(/"/g, '""')}"`
        ).join(',')
      );

      return [headers.join(','), ...rows].join('\n');
    }

    // Для одиночных объектов
    return Object.entries(data)
      .map(([key, value]) => `"${key}","${value}"`)
      .join('\n');
  }

  private static convertToTXT(data: any): string {
    return JSON.stringify(data, null, 2);
  }

  private static validateImportData(
    data: any,
    requiredFields?: string[]
  ): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];

    if (!data) {
      return { valid: false, errors: ['NO_DATA'] };
    }

    if (requiredFields && Array.isArray(requiredFields)) {
      requiredFields.forEach(field => {
        if (!(field in data)) {
          errors.push(`MISSING_FIELD_${field.toUpperCase()}`);
        }
      });
    }

    return {
      valid: errors.length === 0,
      errors: errors.length > 0 ? errors : undefined
    };
  }
}

// Экспортируем экземпляр сервиса для удобства использования
// Удаляем экземпляр, так как все методы статические
