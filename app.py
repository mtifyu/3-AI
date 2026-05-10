import streamlit as st
import os, tempfile, pandas as pd, base64
from dotenv import load_dotenv
from agent import DataAnalystAgent

load_dotenv()
API_KEY = os.getenv("MIMO_API_KEY")

if not API_KEY:
    st.error("API-ключ не найден")
    st.stop()

st.set_page_config(page_title="Задание 3: Мини-продукт с LLM-аналитикой", layout="wide")

st.title("AI Аналитик Данных")
st.markdown("""
*Загрузите файл (CSV/Excel), укажите задачу, и ИИ проведёт анализ автоматически.*
""")

uploaded = st.file_uploader("Загрузите файл", type=['csv', 'xlsx', 'xls'])

instructions = st.text_area(
    "Что проанализировать?",
    placeholder="Пример: Найди корреляции, построй гистограмму возраста, сравни средние по группам...",
    height=80
)

if uploaded and st.button("Запустить анализ", type="primary"):

    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    
    try:
        with st.spinner("ИИ анализирует данные..."):
            agent = DataAnalystAgent(API_KEY)
            
            if instructions and agent.check_prompt_injection(instructions):
                st.error("Подозрительная инструкция. Запрос отклонён.")
                st.stop()

            ok, msg = agent.load_data(tmp_path)
            
            if not ok:
                st.error(f"Ошибка загрузки: {msg}")
            else:
                st.success(f"Файл загружен: {msg}")

                result = agent.analyze_data(instructions)

                if result.get("error"):
                    st.warning(f"Ошибка: {result['error']}")

                st.markdown("Результат анализа:")
                st.markdown(result.get("report", "*Нет данных для отображения*"))
                
                if result.get("plots"):
                    st.markdown("Графики:")
                    for i, plot_b64 in enumerate(result["plots"]):
                        st.image(
                            f"data:image/png;base64,{plot_b64}",
                            caption=f"График {i+1}",
                            use_container_width=True
                        )
                    
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
        with st.expander("Подробности:"):
            st.code(__import__('traceback').format_exc())
            
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)