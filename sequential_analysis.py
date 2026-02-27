"""
работает только с многострочными транскриптами (с таймкодами)
"""
import json
import logging
import time
from typing import Dict, Any, List

from api_client import send_request, poll_request
from config import (
    SYSTEM_PROMPT, POLL_INTERVAL, DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, REASONING_EFFORT
)

logger = logging.getLogger(__name__)

SEQUENTIAL_SYSTEM_PROMPT_FIRST = SYSTEM_PROMPT + """
\n\nЭто первая часть транскрипта. Анализируй только её.
Формируй полный JSON на основе этой части.
"""

SEQUENTIAL_SYSTEM_PROMPT_SUBSEQUENT = SYSTEM_PROMPT + """
\n\nЭто следующая часть транскрипта.
Предыдущий анализ (summary):
{previous_summary}

Интегрируй с предыдущим: обновляй/дополняй поля, суммируй счёты в task_distribution и correct_task,
корректируй speaking_ratio как среднее. Верни обновлённый полный JSON для всего урока.
"""


def split_transcript(transcript: str, num_parts: int = 3) -> List[str]:
    lines = transcript.strip().splitlines()
    if not lines:
        return [""]
    part_size = max(1, len(lines) // num_parts)
    parts = ["\n".join(lines[i:i + part_size]) for i in range(0, len(lines), part_size)]
    while len(parts) > num_parts and len(parts) > 1:
        parts[-2] += "\n" + parts.pop()
    return [p for p in parts if p.strip()] or [""]


def get_summary(result: Dict) -> str:
    if not result:
        return "Предыдущий анализ отсутствует или пустой."
    lines = []
    for k, v in result.items():
        if isinstance(v, dict):
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        elif v and v not in ("Не было", "Не упоминалось"):
            lines.append(f"{k}: {v}")
    return "\n".join(lines) or "Нет значимых данных из предыдущего анализа."


def sequential_analysis(transcript: str, num_parts: int = 3) -> Dict[str, Any]:
    parts = split_transcript(transcript, num_parts)
    if not parts:
        logger.error("Транскрипт пустой после разбиения")
        return {}

    previous_result = None
    for idx, part in enumerate(parts):
        if idx == 0:
            prompt = SEQUENTIAL_SYSTEM_PROMPT_FIRST
            user_text = part
        else:
            summary = get_summary(previous_result)
            prompt_template = SEQUENTIAL_SYSTEM_PROMPT_SUBSEQUENT
            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}").replace("{{previous_summary}}",
                                                                                             "{previous_summary}")
            prompt = escaped_template.format(previous_summary=summary)
            user_text = part

        try:
            request_id = send_request(
                system_prompt=prompt,
                user_text=user_text,
                model=DEFAULT_MODEL,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                reasoning_effort=REASONING_EFFORT
            )
        except Exception as e:
            logger.error(f"Ошибка отправки части {idx + 1}/{len(parts)}: {e}")
            continue

        attempt = 0
        max_attempts = 180  # ~15 мин при 5 сек
        while attempt < max_attempts:
            time.sleep(POLL_INTERVAL)
            res = poll_request(request_id)
            if res is None:
                attempt += 1
                continue

            if res.get("status") == "success":
                logger.info(f"Processing success result for {request_id}")
                raw = res.get("result") or res.get("output") or res.get("text") or res.get("choices") or res.get(
                    "data") or ""
                logger.info(
                    f"Raw type for {request_id}: {type(raw)}, keys: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}, sample: {str(raw)[:200]}")

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
                        logger.warning(f"Could not extract string content from dict raw for {request_id}")
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
                    logger.warning(f"Unexpected raw type {type(raw)} for {request_id}")

                if not content.strip():
                    logger.error(f"Пустой результат для части {idx + 1}")
                    break

                try:
                    parsed = json.loads(content)
                    previous_result = parsed
                except json.JSONDecodeError:
                    logger.error(f"Невалидный JSON в части {idx + 1}:\n{content[:400]}...")
                break

            elif res.get("status") == "failed":
                logger.error(f"Часть {idx + 1} failed: {res}")
                break

            attempt += 1
        else:
            logger.warning(f"Таймаут части {idx + 1}")

    if previous_result is None:
        logger.error("Ни одна часть не дала результата")
        return {}

    return previous_result
