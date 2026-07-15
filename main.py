"""
Main entry point for the keyword extraction system.
Scrapes a webpage, preprocesses content with spaCy, extracts keywords with KeyBERT,
and returns structured JSON output for SEO and content analysis.

Usage:
    python main.py <url>
    python main.py  # interactive prompt
"""

import sys
import io
import json

# Force UTF-8 output on Windows to handle international characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from scraper import scrape_webpage
from preprocessor import preprocess_text
from keyword_extractor import extract_keywords


def analyze_url(url: str) -> dict:
    """
    Full pipeline: scrape -> preprocess -> extract keywords -> structured JSON.

    Steps:
      1. Scrape the webpage for metadata and body content
      2. Preprocess body text with spaCy (tokenization, lemmatization, NER)
      3. Extract primary and secondary keywords with KeyBERT (uses raw text)
      4. Assemble and return structured result
    """

    # --- Step 1: Scrape the webpage ---
    print(f"[1/3] Scraping: {url}")
    scraped = scrape_webpage(url)

    # --- Step 2: Preprocess with spaCy (for entities and content stats) ---
    print("[2/3] Preprocessing text with spaCy...")
    body_preprocessed = preprocess_text(scraped["body_content"])

    # --- Step 3: Extract keywords with KeyBERT (uses raw page text) ---
    print("[3/3] Extracting keywords with KeyBERT...")
    keywords_result = extract_keywords(scraped)

    # --- Assemble final structured output ---
    result = {
        "url": scraped["url"],
        "domain": scraped["domain"],
        "metadata": {
            "title": scraped["metadata"]["title"],
            "description": scraped["metadata"]["description"],
            "keywords": scraped["metadata"]["keywords"],
            "author": scraped["metadata"]["author"],
            "og_type": scraped["metadata"]["og_type"],
            "og_image": scraped["metadata"]["og_image"],
            "canonical_url": scraped["metadata"]["canonical_url"],
        },
        "headings": scraped["headings"],
        "content_stats": {
            "body_length_chars": len(scraped["body_content"]),
            "body_word_count": len(scraped["body_content"].split()),
            "token_count": len(body_preprocessed["tokens"]),
            "unique_token_count": len(set(body_preprocessed["tokens"])),
        },
        "keywords": {
            "primary_keyword": keywords_result["primary_keyword"],
            "secondary_keywords": keywords_result["secondary_keywords"],
        },
        "entities": body_preprocessed["entities"],
        "all_keyword_scores": keywords_result["all_keywords"],
    }

    return result


def main():
    """CLI entry point: accepts a URL from command line or interactive prompt."""
    if len(sys.argv) > 1:
        url = sys.argv[1].strip()
    else:
        url = input("Enter URL to analyze: ").strip()

    if not url:
        print("Error: No URL provided.")
        sys.exit(1)

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        result = analyze_url(url)
        # Pretty-print the JSON output
        print("\n" + "=" * 60)
        print("ANALYSIS RESULT")
        print("=" * 60)
        output = json.dumps(result, indent=2, ensure_ascii=False)
        print(output.encode("utf-8", errors="replace").decode("utf-8"))

    except Exception as e:
        print(f"\nError analyzing URL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
