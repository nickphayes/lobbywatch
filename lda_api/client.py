import time

import requests

from .config import HEADERS
from .utils import get_logger

logger = get_logger(__name__)


def get_with_retries(url, params={}, max_retries=3, backoff=2):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed : {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff**attempt)
            else:
                logger.error(f"Giving up after {max_retries} attempts")
                raise
