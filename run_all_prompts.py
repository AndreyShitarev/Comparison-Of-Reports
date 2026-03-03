# batch_process.py
import os
import time
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from api_client import send_request, poll_request
from config import INPUT_DIR, OUTPUT_DIR      # предполагается, что OUTPUT_DIR есть в config

# Импортируем промпты
from PROMPTS.prompt_1 import SYSTEM_PROMPT as prompt_1
from PROMPTS.prompt_2 import SYSTEM_PROMPT as prompt_2
from PROMPTS.prompt_3 import SYSTEM_PROMPT as prompt_3
from PROMPTS.prompt_4 import SYSTEM_PROMPT as prompt_4
from PROMPTS.prompt_5 import SYSTEM_PROMPT as prompt_5
from PROMPTS.prompt_6 import SYSTEM_PROMPT as prompt_6
from PROMPTS.prompt_7 import SYSTEM_PROMPT as prompt_7
from PROMPTS.prompt_8 import SYSTEM_PROMPT as prompt_8
from PROMPTS.prompt_9 import SYSTEM_PROMPT as prompt_9

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Список всех промптов с их именами (для папок)
PROMPTS_MAP = {
    "prompt_1": prompt_1,
    "prompt_2": prompt_2,
    "prompt_3": prompt_3,
    "prompt_4": prompt_4,
    "prompt_5": prompt_5,
    "prompt_6": prompt_6,
    "prompt_7": prompt_7,
    "prompt_8": prompt_8,
    "prompt_9": prompt_9,
}

# Папка с транскриптами (можно переопределить)
TRANSCRIPTS_DIR = Path("transcripts")
# Максимальное время ожидания одного запроса (в секундах)
MAX_WAIT_SECONDS = 600
# Интервал опроса статуса
POLL_INTERVAL = 6


def read_transcript(path: Path) -> str:
    """Читает весь файл как строку"""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Не удалось прочитать {path}: {e}")
        return ""


def save_result(original_filename: str, full_result: dict, output_subdir: str):
    """Сохраняет результат в подпапку соответствующего промпта"""
    base_dir = Path(OUTPUT_DIR) / output_subdir
    base_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(original_filename).stem
    output_path = base_dir / f"{base_name}.json"

    try:
        if isinstance(full_result.get("result"), list) and full_result["result"]:
            message = full_result["result"][0].get("message", {})
            content_str = message.get("content", "")
            if content_str:
                content_json = json.loads(content_str)
                result_to_save = {"result": content_json}
            else:
                result_to_save = {"result": {}, "error": "Empty content"}
        else:
            result_to_save = {"result": {}, "error": "Unexpected result structure"}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Ошибка парсинга результата {original_filename} → {output_subdir}: {e}")
        result_to_save = {"result": {}, "error": str(e)}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_to_save, f, ensure_ascii=False, indent=4)

    logger.info(f"Сохранено: {output_path}")


def process_one_request(transcript_path: Path, sys_prompt: str, prompt_name: str) -> None:
    """Обрабатывает один файл + один промпт"""
    user_text = read_transcript(transcript_path)
    if not user_text.strip():
        logger.warning(f"Пустой транскрипт, пропускаем: {transcript_path}")
        return

    filename = transcript_path.name
    logger.info(f"→ {filename} | {prompt_name}")

    try:
        request_id = send_request(
            system_prompt=sys_prompt,
            user_text=user_text,
            # можно переопределить модель / температуру / max_tokens если нужно
        )
    except Exception as e:
        logger.error(f"Ошибка отправки {filename} → {prompt_name}: {e}")
        return

    # Пуллинг результата
    start = time.time()
    while time.time() - start < MAX_WAIT_SECONDS:
        try:
            result = poll_request(request_id)
            if result is None:
                time.sleep(POLL_INTERVAL)
                continue
            if result.get("status") == "success":
                save_result(filename, result, prompt_name)
                return
            else:
                logger.error(f"Запрос завершился неуспешно {request_id} → {result}")
                return
        except Exception as e:
            logger.error(f"Ошибка при polling {request_id} ({filename}): {e}")
            time.sleep(POLL_INTERVAL)

    logger.error(f"Таймаут ожидания результата {request_id} ({filename} / {prompt_name})")


def main():
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    if not transcript_files:
        logger.warning(f"Транскрипты в {TRANSCRIPTS_DIR} не найдены")
        return

    logger.info(f"Найдено файлов: {len(transcript_files)}")

    # Параллелим по файлам и промптам → всего 9 × N задач
    tasks = []
    with ThreadPoolExecutor(max_workers=12) as executor:   # подберите под свой лимит API
        for transcript_path in transcript_files:
            for prompt_name, sys_prompt in PROMPTS_MAP.items():
                future = executor.submit(
                    process_one_request,
                    transcript_path,
                    sys_prompt,
                    prompt_name
                )
                tasks.append(future)

        for future in as_completed(tasks):
            try:
                future.result()  # поднимаем исключения, если были
            except Exception as e:
                logger.error(f"Задача упала: {e}")


if __name__ == "__main__":
    main()