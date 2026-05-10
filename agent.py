import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import re
import base64

class DataAnalystAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.df = None
        self.df_info = {}
        
    def load_data(self, file_path: str) -> tuple[bool, str]:
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file_path)
            else:
                return False, "Неподдерживаемый формат"
            
            self._prepare_context()
            return True, f"Загружено: {self.df.shape[0]}×{self.df.shape[1]}"
        except Exception as e:
            return False, f"Ошибка: {e}"
    
    def _prepare_context(self):
        if self.df is None: return
        self.df_info = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "nulls": self.df.isnull().sum().to_dict(),
            "numeric": self.df.select_dtypes(include='number').columns.tolist(),
            "categorical": self.df.select_dtypes(include='object').columns.tolist(),
        }
    
    def _safe_execute(self, code: str):
        result = {"success": False, "output": "", "error": None, "plot_data": None}

        for bad in ['import os', 'import sys', '__import__', 'exec(', 'eval(', 'open(', 'subprocess']:
            if bad in code:
                result["error"] = f"Запрещено: {bad}"
                return result
        
        code = re.sub(r'^(\s*)(import |from )', r'\1# ', code, flags=re.MULTILINE)

        env = {
            "__builtins__": {k: v for k, v in __builtins__.items() if k in [
                'print', 'len', 'range', 'list', 'dict', 'str', 'int', 'float',
                'bool', 'sum', 'min', 'max', 'True', 'False', 'None', 'zip',
                'enumerate', 'sorted', 'round', 'abs', 'repr', 'type'
            ]},
            "pd": pd, "np": np, "plt": plt, "sns": sns,
            "df": self.df.copy() if self.df is not None else None,
        }
        
        old_out = sys.stdout
        sys.stdout = buf = io.StringIO()
        
        try:
            exec(code, env, env)
            result["output"] = buf.getvalue()
            if "final_report" in env and isinstance(env["final_report"], str):
                result["final_report"] = env["final_report"]
            result["success"] = True

            if plt.gcf().get_axes():
                bio = io.BytesIO()
                plt.savefig(bio, format='png', bbox_inches='tight', dpi=100)
                bio.seek(0)
                result["plot_data"] = base64.b64encode(bio.read()).decode()
                plt.close()
                
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = old_out
            plt.close('all')
            
        return result
    
    def _call_llm(self, messages) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Analyst"
        }
        payload = {
            "model": "xiaomi/mimo-v2.5",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4000
        }
        
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    
    def _generate_prompt(self, instructions: str = "") -> str:
        info = self.df_info
        task = instructions or "Проведи EDA: статистика, графики, корреляции, инсайты."
        
        return f"""Ты — data scientist. Сгенерируй ПРОСТОЙ Python-код для анализа.

Данные: {info['shape']}, столбцы: {info['columns'][:10]}{'...' if len(info['columns'])>10 else ''}
Задача: {task}

Можно:
- print(), df.head(), df.describe(), df.corr(numeric_only=True)
- df['col'].value_counts(), df.groupby('col').mean()
- plt.plot(), plt.bar(), plt.hist(), sns.heatmap()

Нельзя:
- import, os, sys, exec, eval, open, subprocess
- sns.boxplot(), sns.violinplot(), сложные groupby().apply()
- plt.show() (график сохранится автоматически)

Правила:
1. Код ТОЛЬКО в блоке ```python ... ```
2. В конце: final_report = "Краткий вывод..."
3. Используй простые конструкции — без вложенных циклов

Пример:
```python
print("Размер:", df.shape)
print(df.describe())
plt.figure(); df['col'].hist(); plt.title('Распределение')
final_report = "Данные полные, среднее по целевой переменной: X" Генерируй ТОЛЬКО код."""
    def _extract_code(self, text: str):
        m = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.I)
        return m.group(1).strip() if m else None

    def analyze_data(self, instructions: str = ""):
        if self.df is None:
            return {"error": "Данные не загружены", "report": "", "plots": []}

        messages = [
            {"role": "system", "content": "Отвечай только кодом в ```python"},
            {"role": "user", "content": self._generate_prompt(instructions)}
        ]
        
        try:
            llm_out = self._call_llm(messages)
        except Exception as e:
            return {"error": f"API: {e}", "report": "", "plots": []}
        
        code = self._extract_code(llm_out)
        if not code:
            return {"error": "LLM не вернул код", "report": llm_out[:500], "plots": []}

        res = self._safe_execute(code)

        report = res.get("final_report") or res["output"] 
        if res.get("error"):
            report += f"\n\n Предупреждение: {res['error']}"

        info = self.df_info 
        return {
            "report": report or "Пустой вывод",
            "plots": [res["plot_data"]] if res.get("plot_data") else [],
            "metadata": {"shape": info["shape"]} if info else None
        }

    def check_prompt_injection(self, text: str) -> bool:
        bad = ['игнорируй', 'ignore', 'забудь', 'forget', 'ты теперь', 'you are now',
            'system:', 'новые инструкции', 'отключи безопасность', 'покажи ключ']
        t = text.lower()
        return any(b in t for b in bad) or len(text) > 2000 or '```' in text