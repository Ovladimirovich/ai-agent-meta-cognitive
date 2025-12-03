export interface ExportData<T = any> {
  version: string;
  timestamp: string;
  type: string;
  data: T;
}

export interface ImportResult<T = any> {
  success: boolean;
  message: string;
  data?: T;
  errors?: string[];
}

export interface ExportOptions {
  format?: 'json' | 'csv' | 'txt';
  includeMetadata?: boolean;
  customFileName?: string;
}

export interface ImportOptions {
  validateSchema?: boolean;
  requiredFields?: string[];
}
