"""
Keyword extraction module using KeyBERT.
Extracts primary keywords (main topic) and secondary keywords (related topics)
with relevance scores for SEO and content analysis.

KeyBERT works on natural language text - it uses BERT embeddings to find
the substrings most similar to the full document. Do NOT pass lemmatized
or heavily preprocessed text here; let KeyBERT handle stop words internally.
"""

from keybert import KeyBERT


# Lazy-load the KeyBERT model
_kw_model = None


def _get_model():
    """Load the KeyBERT model with a lightweight sentence-transformer."""
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _kw_model


def _prepare_text_for_keywords(scraped_data: dict) -> str:
    """
    Assemble the most informative text from the scraped page.
    Focuses on content-rich parts and filters noise.
    """
    parts = []
    metadata = scraped_data.get("metadata", {})
    headings = scraped_data.get("headings", {})

    # Title repeated 3x — strongest keyword signal for SEO
    title = metadata.get("title", "")
    if title:
        parts.extend([title, title, title])

    # Description repeated 2x
    description = metadata.get("description", "")
    if description:
        parts.extend([description, description])

    # Meta keywords if present
    meta_kw = metadata.get("keywords", "")
    if meta_kw:
        parts.append(meta_kw)

    # All headings — they summarize the page structure
    for level in ["h1", "h2", "h3"]:
        for heading in headings.get(level, []):
            parts.append(heading)

    # Body content — take first ~5000 chars (intro + key paragraphs)
    body = scraped_data.get("body_content", "")
    if body:
        # Prioritize the beginning of the article (most SEO-relevant)
        parts.append(body[:5000])

    return " ".join(parts)


def extract_keywords(scraped_data: dict) -> dict:
    """
    Extract keywords from scraped page data using KeyBERT.

    Uses the raw page text (not lemmatized) so BERT embeddings
    capture proper semantic meaning.

    Extracts:
      - Primary keyword: the single most representative keyword (main topic)
      - Secondary keywords: related topics and keyphrases with scores

    Returns a dict with:
        - primary_keyword: dict with keyword and score
        - secondary_keywords: list of dicts with keyword and score
        - all_keywords: full ranked list for reference
    """
    model = _get_model()

    # Build a focused text from the scraped page
    text = _prepare_text_for_keywords(scraped_data)

    if not text or len(text.strip()) < 10:
        return {
            "primary_keyword": {"keyword": "", "score": 0.0},
            "secondary_keywords": [],
            "all_keywords": [],
        }

    # --- Single-word keywords (for primary keyword identification) ---
    single_keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        use_mmr=True,
        diversity=0.3,       # Lower diversity = higher scores for top terms
        top_n=10,
    )

    # --- Multi-word keyphrases (2-3 words) for secondary keywords ---
    keyphrases = model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.5,
        top_n=20,
    )

    # --- Broader extraction (1-3 words) for the full list ---
    broad_keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.6,
        top_n=25,
    )

    # --- Determine primary keyword (highest scoring single word) ---
    primary = {"keyword": "", "score": 0.0}
    if single_keywords:
        best = single_keywords[0]
        primary = {"keyword": best[0], "score": round(best[1], 4)}

    # --- Determine secondary keywords (deduplicated, ranked by score) ---
    seen = set()
    secondary = []

    for kw_tuple in keyphrases + broad_keywords:
        keyword = kw_tuple[0].lower().strip()
        score = round(kw_tuple[1], 4)

        # Skip duplicates, low scores, and the primary keyword
        if keyword in seen or score <= 0:
            continue
        if keyword == primary["keyword"] or primary["keyword"] in keyword:
            continue
        seen.add(keyword)

        secondary.append({"keyword": kw_tuple[0], "score": score})

    # Sort by score descending, take top 10
    secondary.sort(key=lambda x: x["score"], reverse=True)
    secondary = secondary[:10]

    # --- Full ranked list ---
    all_keywords = []
    all_seen = set()
    for kw_tuple in broad_keywords:
        kw = kw_tuple[0]
        if kw.lower() not in all_seen and kw_tuple[1] > 0:
            all_seen.add(kw.lower())
            all_keywords.append({
                "keyword": kw,
                "score": round(kw_tuple[1], 4),
            })

    return {
        "primary_keyword": primary,
        "secondary_keywords": secondary,
        "all_keywords": all_keywords,
    }
