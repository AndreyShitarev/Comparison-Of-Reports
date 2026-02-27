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
INPUT_DIR = "transcripts"


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

Оба — JSON строго фиксированной структуры.

Оцени степень совпадения CANDIDATE с ETALON по следующим весам (сумма = 1.0):

task_distribution          → 0.30  (очень важный блок)
correct_task               → 0.22
числовые соотношения       → 0.13  (в т.ч. speaking_ratio)
homework + school_topic    → 0.08
brief_review + theory_detail + practice → 0.10
independent_results + reflection_summary → 0.07
остальные текстовые поля   → 0.10

Правила оценки числовых полей (особенно важно!):

1. task_distribution и correct_task — **относительная ошибка**, а не абсолютная
   score = 1 - min( |a-et| / max(a,et,1) , 1.0 )   для каждого из 4-х значений
   итого — среднее по четырём

2. correct_task.* не может быть больше соответствующего task_distribution.*

3. speaking_ratio — тоже относительная ошибка (допустимо ±15–20% без сильного штрафа)

Шкала итоговой оценки (после взвешивания):

1.00 — почти идентично
0.90–0.99 — отличное, мелкие расхождения
0.75–0.89 — хорошее, заметные, но не критичные отличия
0.55–0.74 — среднее, уже есть системные ошибки
0.30–0.54 — слабое соответствие
0.00–0.29 — почти не соответствует / серьёзные логические/числовые ошибки

Верни ТОЛЬКО одно число от 0.00 до 1.00 с двумя знаками после точки.
Никакого текста, никаких пояснений.
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


    return avg


if __name__ == "__main__":
    folders = [
        "output/base_analysis",
        "output/parallel_analysis",
        "output/sequential_analysis",
    ]

    for folder in folders:
        score = main(folder)
        if score is not None:
            print(f"{folder:<25} → {score:.3f}")
        else:
            print(f"{folder:<25} → ошибка")