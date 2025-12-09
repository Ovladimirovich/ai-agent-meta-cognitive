
// Типы для уведомлений
export interface NotificationOptions {
  title: string;
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
}

// Глобальный список уведомлений
let notifications: NotificationOptions[] = [];

// Функция для отображения уведомления
export const showNotification = (options: NotificationOptions) => {
  const { title, message, type = 'info', duration = 5000 } = options;

  // В реальной реализации здесь будет логика отображения уведомления
  // Пока что просто выводим в консоль
  console.log(`[${type.toUpperCase()}] ${title}: ${message}`);

  // Создаем объект уведомления
  const notification: NotificationOptions = {
    title,
    message,
    type,
    duration
  };

  // Добавляем в глобальный список
  notifications.push(notification);

  // Удаляем уведомление через определенное время
  setTimeout(() => {
    notifications = notifications.filter(n => n !== notification);
  }, duration);
};

// Функция для получения всех уведомлений
export const getNotifications = (): NotificationOptions[] => {
  return [...notifications];
};

// Функция для очистки уведомлений
export const clearNotifications = () => {
  notifications = [];
};
