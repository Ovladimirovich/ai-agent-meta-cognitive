/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  // другие переменные окружения, если есть
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}