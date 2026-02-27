import os
import json
import glob
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GENAPI_API_KEY")
BASE_URL = "https://api.gen-api.ru"
MODEL = "deepseek-chat"

ENDPOINT = f"{BASE_URL}/api/v1/networks/{MODEL}"

ETALON_DIR = "etalons"
INPUT_DIR = "input"


def load_jsons(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    files = sorted(glob.glob(os.path.join(folder, "*.json")))

    if not files:
        raise ValueError(f"В папке {folder} нет JSON файлов")

    data = []

    for f in files:
        try:
            with open(f, encoding="utf-8") as file:
                data.append(json.load(file))
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка JSON в файле {f}: {e}")
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения файла {f}: {e}")

    return data


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
    try:
        etalon_str = json.dumps(etalon, ensure_ascii=False)
        candidate_str = json.dumps(candidate, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Ошибка сериализации JSON: {e}")

    return f"""
ETALON:
{etalon_str}

CANDIDATE:
{candidate_str}

Верни итоговую оценку.
""".strip()


def ask_llm(prompt):
    if not API_KEY:
        raise EnvironmentError("Не задан GENAPI_API_KEY")

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

    try:
        r = requests.post(
            ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise ConnectionError(f"Ошибка запроса к API: {e}")

    if r.status_code != 200:
        raise RuntimeError(
            f"API вернул статус {r.status_code}: {r.text}"
        )

    try:
        data = r.json()
    except json.JSONDecodeError:
        raise ValueError("Ответ API не является валидным JSON")

    try:
        content = data["response"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise ValueError("Неожиданная структура ответа API")

    try:
        score = float(content)
    except ValueError:
        raise ValueError(f"Модель вернула не число: {content}")

    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Оценка вне диапазона 0–1: {score}")

    return score


def main(input_dir: str = INPUT_DIR):
    try:
        etalons = load_jsons(ETALON_DIR)
        candidates = load_jsons(input_dir)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return

    if len(etalons) != len(candidates):
        print(
            f"Количество эталонов ({len(etalons)}) "
            f"и кандидатов ({len(candidates)}) не совпадает"
        )
        return

    scores = []

    for idx, (e, c) in enumerate(zip(etalons, candidates), 1):
        try:
            prompt = build_prompt(e, c)
            score = ask_llm(prompt)
            scores.append(score)
        except Exception as e:
            print(f"Ошибка при обработке пары #{idx}: {e}")
            continue

    if not scores:
        print("Нет валидных оценок для расчёта среднего")
        return

    avg = sum(scores) / len(scores)

    print(input_dir, avg)


if __name__ == "__main__":
    main()
