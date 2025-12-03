import React, { useRef, useState } from 'react';
import { importExportService } from '@/shared/importExportService';
import { Button } from '@/shared/ui/Button';
import { IconFileImport, IconFileExport } from '@tabler/icons-react';
import { showNotification } from '@/shared/ui/Notifications';

interface ImportExportButtonsProps<T = any> {
  /** Тип данных для экспорта/импорта (используется в имени файла) */
  dataType: string;
  /** Данные для экспорта */
  exportData?: T;
  /** Обработчик успешного импорта */
  onImport?: (data: T) => void;
  /** Дополнительные настройки экспорта */
  exportOptions?: {
    format?: 'json' | 'csv' | 'txt';
    fileName?: string;
  };
  /** Валидация при импорте */
  importOptions?: {
    requiredFields?: string[];
    validate?: (data: any) => { valid: boolean; error?: string };
  };
  /** Текст для кнопки импорта */
  importText?: string;
  /** Текст для кнопки экспорта */
  exportText?: string;
  /** Классы для стилизации контейнера */
  className?: string;
}

export function ImportExportButtons<T = any>({
  dataType,
  exportData,
  onImport,
  exportOptions = {},
  importOptions = {},
  importText = 'Импорт',
  exportText = 'Экспорт',
  className = '',
}: ImportExportButtonsProps<T>) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleExport = async () => {
    if (!exportData) {
      showNotification({
        title: 'Ошибка',
        message: 'Нет данных для экспорта',
        type: 'error',
      });
      return;
    }

    try {
      setIsLoading(true);
      const success = await importExportService.exportData(
        exportData,
        dataType,
        {
          format: exportOptions.format || 'json',
          customFileName: exportOptions.fileName,
        }
      );

      if (success) {
        showNotification({
          title: 'Успех',
          message: 'Данные успешно экспортированы',
          type: 'success',
        });
      }
    } catch (error) {
      console.error('Export error:', error);
      showNotification({
        title: 'Ошибка',
        message: 'Не удалось экспортировать данные',
        type: 'error',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsLoading(true);
      const result = await importExportService.importData<T>(file, {
        validateSchema: true,
        requiredFields: importOptions.requiredFields,
      });

      if (result.success && result.data) {
        // Дополнительная валидация, если предоставлена
        if (importOptions.validate) {
          const validation = importOptions.validate(result.data);
          if (!validation.valid) {
            throw new Error(validation.error || 'Ошибка валидации данных');
          }
        }

        onImport?.(result.data);
        
        showNotification({
          title: 'Успех',
          message: 'Данные успешно импортированы',
          type: 'success',
        });
      } else {
        throw new Error(result.message || 'Неизвестная ошибка при импорте');
      }
    } catch (error: any) {
      console.error('Import error:', error);
      showNotification({
        title: 'Ошибка импорта',
        message: error.message || 'Не удалось импортировать данные',
        type: 'error',
      });
    } finally {
      // Сбрасываем значение input, чтобы можно было загрузить тот же файл снова
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      setIsLoading(false);
    }
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {/* Скрытый input для загрузки файла */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".json,.csv,.txt"
        className="hidden"
        disabled={isLoading}
      />

      {/* Кнопка импорта */}
      <Button
        variant="outline"
        onClick={() => fileInputRef.current?.click()}
        disabled={isLoading}
        leftIcon={<IconFileImport size={16} />}
      >
        {importText}
      </Button>

      {/* Кнопка экспорта */}
      {exportData && (
        <Button
          variant="outline"
          onClick={handleExport}
          disabled={isLoading || !exportData}
          leftIcon={<IconFileExport size={16} />}
        >
          {exportText}
        </Button>
      )}
    </div>
  );
}

export default ImportExportButtons;
