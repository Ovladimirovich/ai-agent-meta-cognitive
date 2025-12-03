def validate_content(self, text: str) -> Dict[str, Any]:
        """Комплексная валидация контента"""
        result = {
            'is_valid': True,
            'is_safe': True,
            'risk_level': 'low',
            'checks': {},
            'issues': [],
            'recommendations': []
        }

        # Проверка длины
        length_check = self.validate_input_length(text)
        result['checks']['length'] = length_check
        if not length_check['is_valid']:
            result['is_valid'] = False
            result['is_safe'] = False
            result['risk_level'] = 'medium'
            result['issues'].extend(length_check['issues'])
            result['recommendations'].extend(length_check['recommendations'])

        # Проверка HTML/XSS
        html_check = self._check_html_content(text)
        result['checks']['html'] = html_check
        if not html_check['is_safe']:
            result['is_valid'] = False
            result['is_safe'] = False
            result['risk_level'] = 'high'
            result['issues'].extend(html_check['issues'])
            result['recommendations'].extend(html_check['recommendations'])

        # Проверка SQL инъекций
        sql_check = self.validate_sql_injection(text)
        result['checks']['sql'] = sql_check
        if not sql_check['is_safe']:
            result['is_valid'] = False
            result['is_safe'] = False
            result['risk_level'] = max(result['risk_level'], sql_check['risk_level'])
            result['issues'].extend([f"Обнаружены подозрительные SQL паттерны: {', '.join(sql_check['detected_patterns'])}"])
            result['recommendations'].extend(sql_check['recommendations'])

        return result
