from lda_api.fetch import fetch_filings

if __name__ == "__main__":
    filings = fetch_filings(year=2024, period="first_quarter", max_records=2000)
