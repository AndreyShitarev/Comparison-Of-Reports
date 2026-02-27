import time
import requests
import logging
from typing import Optional, Dict, Any
from config import API_KEY, CREATE_URL, STATUS_URL, HEADERS, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, REASONING_EFFORT

logger = logging.getLogger(__name__)


def send_request(
    system_prompt: str,
    user_text: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    reasoning_effort: str = REASONING_EFFORT
) -> str:
    """Send asynchronous request to the API and return request_id."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "top_k": 0,
        "is_sync": False,
        "reasoning_effort": reasoning_effort
    }

    headers = {**HEADERS, "Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.post(CREATE_URL, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        request_id = data.get("request_id")

        if not request_id:
            raise RuntimeError(f"No request_id in response: {data}")

        logger.info(f"Submitted request, got ID: {request_id}")
        return str(request_id)

    except requests.RequestException as e:
        logger.error(f"Failed to send request: {e}")
        raise


def poll_request(request_id: str) -> Optional[Dict[str, Any]]:
    """Poll request status. Returns full response on success, None if pending, raises on failure."""
    headers = {**HEADERS, "Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.get(f"{STATUS_URL}/{request_id}", headers=headers, timeout=300)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")

        if status == "success":
            logger.info(f"Request {request_id} completed successfully")
            return data
        if status == "failed":
            logger.error(f"Request {request_id} failed: {data}")
            raise RuntimeError(f"Request failed: {data}")
        logger.debug(f"Request {request_id} still pending")
        return None

    except requests.RequestException as e:
        logger.error(f"Error polling request {request_id}: {e}")
        raise