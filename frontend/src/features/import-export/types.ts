export type ExportData = {
  version: string;
  timestamp: string;
  data: {
    [key: string]: any;
  };
};

export type ImportResult = {
  success: boolean;
  message: string;
  data?: any;
  errors?: string[];
};

export type ExportOptions = {
  includeSettings: boolean;
  includeHistory: boolean;
  includeModels: boolean;
  customData?: Record<string, any>;
};

export const DEFAULT_EXPORT_OPTIONS: ExportOptions = {
  includeSettings: true,
  includeHistory: true,
  includeModels: true,
};
