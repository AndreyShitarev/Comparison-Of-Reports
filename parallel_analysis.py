"""
работает только с многострочными транскриптами (с таймкодами)
"""
import json
import logging
import time
from typing import List, Dict, Any

from api_client import send_request, poll_request
from config import (
    SYSTEM_PROMPT,
    POLL_INTERVAL,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    REASONING_EFFORT
)

logger = logging.getLogger(__name__)

PARALLEL_SYSTEM_PROMPT = SYSTEM_PROMPT + """
\n\nТы анализируешь только предоставленную часть транскрипта.
Формируй JSON только для этой части, игнорируя полноту урока.
Не добавляй информацию за пределами части.
Если часть неполная, укажи 'Не упоминалось' или 'Не было' для отсутствующих пунктов.
"""


def split_transcript(transcript: str, num_parts: int = 3) -> List[str]:
    """Разбивает транскрипт на примерно равные части по строкам."""
    lines = transcript.strip().splitlines()
    if not lines:
        return [""]

    part_size = max(1, len(lines) // num_parts)
    parts = []
    for i in range(0, len(lines), part_size):
        part = "\n".join(lines[i:i + part_size])
        if part.strip():
            parts.append(part)

    # Если последняя часть слишком маленькая — присоединяем к предпоследней
    while len(parts) > num_parts and len(parts) > 1:
        parts[-2] = parts[-2] + "\n" + parts.pop()

    return parts or [""]


def parallel_analysis(transcript: str, num_parts: int = 3) -> Dict[str, Any]:
    parts = split_transcript(transcript, num_parts)
    if not parts:
        return {}

    request_ids = []
    for part in parts:
        try:
            rid = send_request(
                system_prompt=PARALLEL_SYSTEM_PROMPT,
                user_text=part,
                model=DEFAULT_MODEL,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                reasoning_effort=REASONING_EFFORT
            )
            request_ids.append(rid)
        except Exception as e:
            logger.error(f"Ошибка отправки части: {e}")
            continue

    if not request_ids:
        logger.error("Не удалось отправить ни одну часть на анализ")
        return {}

    results = []
    for rid in request_ids:
        attempt = 0
        max_attempts = 120  # ~10 минут при POLL_INTERVAL=5
        while attempt < max_attempts:
            time.sleep(POLL_INTERVAL)
            res = poll_request(rid)
            if res is None:
                attempt += 1
                continue

            if res.get("status") == "success":
                logger.info(f"Processing success result for {rid}")
                raw = res.get("result") or res.get("output") or res.get("text") or res.get("choices") or res.get(
                    "data") or ""
                logger.info(
                    f"Raw type for {rid}: {type(raw)}, keys: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}, sample: {str(raw)[:200]}")

                content = None
                if isinstance(raw, str):
                    content = raw
                elif isinstance(raw, dict):
                    # Try various paths to extract content
                    paths = [
                        lambda d: d.get("choices", [{}])[0].get("message", {}).get("content"),
                        lambda d: d.get("choices", [{}])[0].get("text"),
                        lambda d: d.get("content"),
                        lambda d: d.get("message", {}).get("content"),
                        lambda d: d.get("output", {}).get("text"),
                    ]
                    for path in paths:
                        try:
                            c = path(raw)
                            if isinstance(c, str) and c.strip():
                                content = c
                                break
                        except Exception:
                            pass
                    if content is None:
                        logger.warning(f"Could not extract string content from dict raw for {rid}")
                        content = ""
                elif isinstance(raw, list):
                    try:
                        first = raw[0]
                        if isinstance(first, str):
                            content = first
                        elif isinstance(first, dict):
                            content = first.get("content") or first.get("text") or first.get("message", {}).get(
                                "content")
                    except IndexError:
                        pass
                    if content is None:
                        content = ""
                else:
                    content = ""
                    logger.warning(f"Unexpected raw type {type(raw)} for {rid}")

                if not content:
                    logger.error(f"Пустой результат для {rid}")
                    break

                try:
                    parsed = json.loads(content)
                    results.append(parsed)
                except json.JSONDecodeError:
                    logger.error(f"Невалидный JSON в ответе {rid}:\n{content[:300]}...")
                break

            elif res.get("status") == "failed":
                logger.error(f"Запрос {rid} завершился с ошибкой: {res}")
                break

            attempt += 1
        else:
            logger.warning(f"Таймаут ожидания результата для {rid}")

    if not results:
        logger.error("Ни одна часть не вернула валидный результат")
        return {}

    return merge_results(results)


def merge_results(results: List[Dict]) -> Dict[str, Any]:
    """Сливает частичные результаты в один отчёт."""
    if not results:
        return {}

    merged = {
        "homework": [],
        "school_topic": [],
        "questions": [],
        "prev_grades": [],
        "upcoming_tests": [],
        "mood_check": [],
        "goals_clarity": [],
        "brief_review": [],
        "theory_detail": [],
        "practice": [],
        "task_distribution": {"teacher_only": 0, "together": 0, "student_independent": 0, "student_silent": 0},
        "correct_task": {"together": 0, "student_independent": 0, "student_silent": 0},
        "independent_results": [],
        "method_flexibility": [],
        "explanation_clarity": [],
        "student_engagement": [],
        "feedback_check": [],
        "safe_environment": [],
        "reflection_summary": [],
        "speaking_ratio": []
    }

    for res in results:
        for key in [
            "homework", "school_topic", "questions", "prev_grades", "upcoming_tests",
            "mood_check", "goals_clarity", "brief_review", "theory_detail", "practice",
            "independent_results", "method_flexibility", "explanation_clarity",
            "student_engagement", "feedback_check", "safe_environment", "reflection_summary"
        ]:
            val = res.get(key)
            if val and val not in ("Не было", "Не упоминалось", ""):
                if isinstance(val, str):
                    merged[key].append(val.strip())
                elif isinstance(val, list):
                    merged[key].extend([v.strip() for v in val if v.strip()])

        # Счётчики задач
        dist = res.get("task_distribution", {})
        for k in merged["task_distribution"]:
            merged["task_distribution"][k] += int(dist.get(k, 0))

        corr = res.get("correct_task", {})
        for k in merged["correct_task"]:
            merged["correct_task"][k] += int(corr.get(k, 0))

        sr = res.get("speaking_ratio")
        if sr and isinstance(sr, str) and "/" in sr:
            merged["speaking_ratio"].append(sr)

    # Финальная обработка
    for key in list(merged):
        if isinstance(merged[key], list) and key != "speaking_ratio":
            unique = []
            seen = set()
            for item in merged[key]:
                if item not in seen:
                    unique.append(item)
                    seen.add(item)
            merged[key] = "; ".join(unique) if unique else "Не упоминалось"

    if merged["speaking_ratio"]:
        ratios = []
        for r in merged["speaking_ratio"]:
            try:
                teacher = int(r.split("/")[0].strip("% "))
                ratios.append(teacher)
            except:
                pass
        if ratios:
            avg = round(sum(ratios) / len(ratios))
            merged["speaking_ratio"] = f"{avg}%/{100 - avg}%"
        else:
            merged["speaking_ratio"] = "Не упоминалось"
    else:
        merged["speaking_ratio"] = "Не упоминалось"

    # Проверка целостности
    total_tasks = sum(merged["task_distribution"].values())
    total_correct = sum(merged["correct_task"].values())
    if total_correct > total_tasks:
        logger.warning(f"Несоответствие: correct {total_correct} > total {total_tasks}")

    return merged
