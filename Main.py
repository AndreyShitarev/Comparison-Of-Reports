#!/usr/bin/env python3

import os
import json
import glob
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GENAPI_API_KEY")
BASE_URL = "https://api.gen-api.ru"
MODEL = "deepseek-chat" # Дима, это нейронка, которую ты предложил deepseek v.3.2

ENDPOINT = f"{BASE_URL}/api/v1/networks/{MODEL}"

ETALON_DIR = "etalons"
INPUT_DIR = "input"


def load_jsons(folder):
    files = sorted(glob.glob(f"{folder}/*.json"))
    return [json.load(open(f, encoding="utf-8")) for f in files]


SYSTEM_PROMPT = """
Ты — эксперт-ассессор качества аналитических отчётов онлайн-уроков.

Тебе даются:
1) ЭТАЛОННЫЙ отчёт (идеальный)
2) CANDIDATE отчёт (созданный нейросетью)

Оба являются JSON строго фиксированной структуры.

Твоя задача — вычислить степень совпадения CANDIDATE с ETALON.

Оценка должна учитывать:

1. Структурное совпадение
- наличие всех ключей
- отсутствие пропусков
- корректные типы данных

2. Числовые поля (КРИТИЧЕСКИ ВАЖНО)
- task_distribution сравнивается строго
- correct_task не может превышать task_distribution
- любые числовые расхождения существенно снижают оценку

3. Семантическое совпадение текстовых полей
- одинаковый смысл → высокий балл
- частичное совпадение → средний
- общий, размытый или неверный анализ → сильный штраф

4. Следование регламенту анализа
- отсутствие выдуманных фактов
- отсутствие пропущенных пунктов
- корректность формулировок
- объективность

5. Штрафы
- отсутствующее поле → сильный штраф
- логические противоречия → сильный штраф
- неверные числовые выводы → сильный штраф

6. Интерпретация оценки
1.0 — практически идентичный отчёт  
0.8–0.95 — отличное совпадение, minor различия  
0.6–0.8 — заметные отличия  
0.4–0.6 — частично корректный анализ  
0.2–0.4 — слабое соответствие  
0.0–0.2 — анализ неверный

ВАЖНО:
- Верни ТОЛЬКО одно число от 0 до 1.
- Никакого текста.
- Никаких пояснений.
- Только число.
""".strip()


def build_prompt(etalon, candidate):
    return f"""
ETALON:
{json.dumps(etalon, ensure_ascii=False)}

CANDIDATE:
{json.dumps(candidate, ensure_ascii=False)}

Верни итоговую оценку.
""".strip()


def ask_llm(prompt):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "is_sync": True,
    }

    r = requests.post(ENDPOINT, headers=headers, json=payload)
    data = r.json()
    return float(data["response"][0]["message"]["content"].strip())


def main():

    etalons = load_jsons(ETALON_DIR)
    candidates = load_jsons(INPUT_DIR)

    scores = []

    for e, c in zip(etalons, candidates):
        prompt = build_prompt(e, c)
        score = ask_llm(prompt)
        scores.append(score)

    avg = sum(scores) / len(scores)

    print(avg)


if __name__ == "__main__":
    main()
