import json
import os

from .client import get_with_retries
from .config import BASE_URL, DEFAULT_PAGE_SIZE
from .utils import get_logger

logger = get_logger(__name__)


def fetch_filings(
    year=2024, period="first_quarter", max_records=2000, save_dir="data/raw"
):
    os.makedirs(save_dir, exist_ok=True)

    page = 1
    all_filings = []

    while len(all_filings) < max_records:
        params = {
            "filing_year": year,
            "filing_period": period,
            "page": page,
            "page_size": DEFAULT_PAGE_SIZE,
        }

        data = get_with_retries(f"{BASE_URL}/filings", params=params)
        results = data.get("results", [])

        if not results:
            logger.info("No results on page, stopping")
            break  # out of pages

        all_filings.extend(results)
        logger.info(f"Fetched page {page}: total {len(all_filings)} filings")

        if len(all_filings) >= max_records:
            logger.info(f"Reached max records: {max_records}")
            break

        page += 1

    with open(os.path.join(save_dir, "filings.json"), "w") as f:
        json.dump(all_filings, f, indent=2)
    logger.info("Results saved to filings.json")
    return all_filings
