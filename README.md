# Keyword Extraction System for SEO & Content Analysis

A Python tool that accepts any webpage URL, scrapes its content, and extracts SEO-ready keywords with relevance scores using spaCy and KeyBERT.

---

## What This Tool Does

Give it a URL and it returns a structured JSON report containing:

- **Page metadata** — title, description, Open Graph tags, canonical URL, author
- **Headings** — all h1 through h6 tags grouped by level
- **Content stats** — character count, word count, unique token count
- **Primary keyword** — the single most representative word for the page
- **Secondary keywords** — 10 related keyphrases with cosine similarity scores
- **Named entities** — people, organizations, locations, dates detected in the text
- **Full keyword rankings** — complete list of extracted keywords with scores

---

## Why Each Decision Was Made

### Why spaCy for preprocessing and not NLTK or pure regex?

spaCy provides an integrated pipeline: tokenization, lemmatization, POS tagging, dependency parsing, and NER all run in a single `nlp()` call. NLTK requires separate calls for each step and is slower on large texts. Regex-based tokenizers break on edge cases like "U.K.", "don't", or hyphenated words. spaCy handles these correctly out of the box with language-specific rules.

### Why KeyBERT and not RAKE, YAKE, or TF-IDF?

RAKE and YAKE are statistical — they rank words by frequency and co-occurrence patterns. They often extract common but meaningless words like "method" or "approach". TF-IDF requires a corpus of documents to compute inverse document frequency. KeyBERT uses BERT embeddings and cosine similarity: it finds the substrings whose semantic meaning is closest to the document's overall meaning. This produces keywords that actually describe the topic, not just the most frequent words.

### Why not lemmatize text before KeyBERT?

BERT models are pre-trained on natural language. When you lemmatize ("running" → "run", "technologies" → "technology"), you destroy the contextual embeddings that BERT relies on. The cosine similarity scores drop and the extracted keywords become garbled phrases. KeyBERT handles stop word removal internally via its `stop_words` parameter. The preprocessing module (spaCy) is used **only** for NER and content statistics, not for feeding into KeyBERT.

### Why repeat the title 3x and description 2x for KeyBERT?

KeyBERT finds substrings most similar to the full document. On a long page, the title's signal gets diluted across thousands of words. Repeating it increases its weight in the document embedding, pulling the keyword extraction toward the actual topic. This mirrors how search engines treat `<title>` as the strongest relevance signal.

### Why truncate body content to 5000 characters?

Wikipedia articles and news pages can be 50,000+ characters. KeyBERT computes embeddings for every candidate n-gram, which becomes very slow on long texts. The first 5000 characters of a page typically contain the introduction and key paragraphs — the most SEO-relevant content. The title and headings (which are included separately) cover the rest.

### Why MMR (Maximal Marginal Relevance)?

Without diversity control, KeyBERT returns "machine learning", "learning machine", "machine learning algorithm" — semantically identical results. MMR adds a diversity penalty: each new keyword must be dissimilar to already-selected keywords. A `diversity=0.3` for the primary keyword keeps scores high (favors relevance), while `diversity=0.5-0.6` for secondary keywords ensures variety.

---

## Project Structure

```
Keyword ES/
├── main.py                 # Entry point — orchestrates the pipeline
├── scraper.py              # Web scraping — fetches HTML, extracts metadata and body text
├── preprocessor.py         # spaCy NLP — tokenization, lemmatization, NER
├── keyword_extractor.py    # KeyBERT — keyword and keyphrase extraction with scores
└── requirements.txt        # Python dependencies
```

### `main.py` — Pipeline Orchestrator

Runs three steps in sequence:

1. Calls `scrape_webpage(url)` to get raw page data
2. Calls `preprocess_text(body)` to get tokens and named entities via spaCy
3. Calls `extract_keywords(scraped_data)` to get keywords via KeyBERT

Assembles everything into the final JSON structure. Forces UTF-8 output on Windows to handle international characters (e.g., accented names, CJK text).

### `scraper.py` — Web Content Extraction

| Function | What It Does |
|---|---|
| `fetch_page(url)` | Downloads HTML with a browser-like User-Agent header. Uses `requests` with a 15-second timeout. |
| `extract_metadata(soup)` | Pulls `<title>`, `og:title`, `og:description`, meta description, meta keywords, `og:type`, `og:image`, canonical URL, and author from the HTML. Prefers Open Graph tags over standard meta tags. |
| `extract_headings(soup)` | Collects all h1–h6 tags into a dict grouped by level. |
| `_remove_boilerplate(soup)` | Strips scripts, styles, nav bars, footers, ads, cookie banners, modals, hidden elements, and HTML comments. Uses a combination of tag names and CSS selectors. |
| `extract_body_content(soup)` | Finds the main content area (`<main>`, `<article>`, or `role="main"`), applies boilerplate removal, converts `<br>` and `<p>` to newlines, and returns clean text. |

**Why BeautifulSoup + requests and not Selenium or Playwright?**
This tool extracts metadata and static body content. It does not need to execute JavaScript, handle SPAs, or interact with pages. `requests` + `BeautifulSoup` is simpler, faster, and has no browser dependency.

**Why lxml parser?**
`lxml` is a C-based HTML parser that is significantly faster than Python's built-in `html.parser`. It also handles malformed HTML more gracefully, which matters when scraping diverse websites.

### `preprocessor.py` — spaCy Text Processing

| What | How |
|---|---|
| **Tokenization** | spaCy's language-specific tokenizer splits text into tokens, handling abbreviations ("U.K."), contractions ("don't"), and punctuation correctly. |
| **Lemmatization** | Uses spaCy's rule-based lemmatizer to reduce words to base forms ("running" → "run", "better" → "good"). |
| **Stop word removal** | Filters tokens marked as stop words by spaCy's English stop list (words like "the", "is", "at" that carry no topic signal). |
| **Named Entity Recognition** | spaCy's `en_core_web_sm` model labels spans as PERSON, ORG, GPE, DATE, MONEY, etc. Entities are deduplicated by lowercase text. |
| **POS tagging** | Each token gets a part-of-speech tag (NOUN, VERB, ADJ, etc.) returned as `(token, tag)` tuples. |

**Why `en_core_web_sm` and not `en_core_web_lg`?**
The small model (~12MB) is fast, requires no extra downloads, and provides adequate NER for most English content. The large model adds word vectors and higher accuracy but is 560MB and slower to load. You can swap it by changing one line in `_get_nlp()`.

**Why lazy-load the model?**
spaCy model loading takes 1-2 seconds. Lazy loading avoids this penalty if the module is imported but not used, and ensures the model is loaded only once per session.

### `keyword_extractor.py` — KeyBERT Keyword Extraction

| Parameter | Value | Why |
|---|---|---|
| `model` | `all-MiniLM-L6-v2` | Lightweight sentence-transformer (80MB). Good balance of speed and quality for English text. |
| `keyphrase_ngram_range` | `(1,1)` for primary, `(2,3)` for secondary | Single words identify the main topic. 2-3 word phrases capture related concepts. |
| `stop_words` | `"english"` | KeyBERT's CountVectorizer filters English stop words from candidate generation. |
| `use_mmr` | `True` | Enables Maximal Marginal Relevance to avoid redundant keywords. |
| `diversity` | `0.3` primary, `0.5` secondary, `0.6` full list | Lower diversity = higher scores for top terms. Higher diversity = more varied results. |
| `top_n` | `10` primary, `20` phrases, `25` full | Enough candidates to filter and rank without overwhelming the output. |

**Text preparation strategy (`_prepare_text_for_keywords`):**
1. Title repeated 3x — strongest keyword signal
2. Description repeated 2x — reinforces topic
3. Meta keywords if present
4. All h1, h2, h3 headings — page structure summary
5. First 5000 characters of body — intro and key content

This focused text goes to KeyBERT instead of the full 50,000+ character page, producing faster and more accurate results.

---

## Setup

### Prerequisites

- Python 3.10 or higher
- `uv` package manager (recommended) or `pip`

### Installation

```bash
# Clone or download the project
cd Keyword ES

# Create a virtual environment (if not already created)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download spaCy English model
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Or with pip:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `requests` | >=2.31.0 | HTTP requests to download web pages |
| `beautifulsoup4` | >=4.12.0 | HTML parsing and content extraction |
| `lxml` | >=4.9.0 | Fast C-based HTML parser for BeautifulSoup |
| `spacy` | >=3.7.0 | NLP pipeline: tokenization, lemmatization, NER |
| `keybert` | >=0.8.0 | BERT-based keyword extraction |
| `en_core_web_sm` | 3.8.0 | spaCy's small English model (NER, POS, lemmatizer) |

KeyBERT automatically installs `sentence-transformers`, `torch`, `transformers`, and `scikit-learn` as its own dependencies.

---

## Usage

### Command Line

```bash
# Pass URL as argument
python main.py "https://en.wikipedia.org/wiki/Artificial_intelligence"

# Interactive prompt (asked when no argument is provided)
python main.py
# Then enter the URL when prompted
```

### As a Python Module

```python
from main import analyze_url

result = analyze_url("https://www.google.com/")

# Access specific parts
print(result["keywords"]["primary_keyword"])
# {'keyword': 'google', 'score': 0.5277}

print(result["keywords"]["secondary_keywords"])
# [{'keyword': 'uiux design projects', 'score': 0.4645}, ...]

print(result["entities"][:3])
# [{'text': 'UI / UX Design Building', 'label': 'ORG', ...}, ...]
```

---

## Output Format

```json
{
  "url": "https://example.com",
  "domain": "example.com",
  "metadata": {
    "title": "Page Title",
    "description": "Meta description text",
    "keywords": "comma, separated, keywords",
    "author": "Author Name",
    "og_type": "website",
    "og_image": "https://example.com/image.jpg",
    "canonical_url": "https://example.com"
  },
  "headings": {
    "h1": ["Main Heading"],
    "h2": ["Section 1", "Section 2"],
    "h3": ["Subsection 1"]
  },
  "content_stats": {
    "body_length_chars": 12345,
    "body_word_count": 2100,
    "token_count": 1400,
    "unique_token_count": 650
  },
  "keywords": {
    "primary_keyword": {
      "keyword": "machine learning",
      "score": 0.6201
    },
    "secondary_keywords": [
      { "keyword": "artificial intelligence", "score": 0.5834 },
      { "keyword": "neural networks", "score": 0.4512 },
      { "keyword": "deep learning", "score": 0.4102 }
    ]
  },
  "entities": [
    {
      "text": "Google",
      "label": "ORG",
      "description": "Companies, agencies, institutions, etc.",
      "start": 150,
      "end": 156
    }
  ],
  "all_keyword_scores": [
    { "keyword": "machine learning", "score": 0.6201 },
    { "keyword": "artificial intelligence", "score": 0.5834 }
  ]
}
```

### Field Descriptions

| Field | Description |
|---|---|
| `metadata.title` | Page title from `og:title` or `<title>` tag |
| `metadata.description` | From `og:description` or `<meta name="description">` |
| `metadata.keywords` | From `<meta name="keywords">` (often empty, many sites don't use this) |
| `metadata.canonical_url` | The preferred URL for this page (SEO canonical) |
| `headings` | All heading tags grouped by level — reveals page structure |
| `content_stats.token_count` | spaCy tokens after lemmatization and stop word removal |
| `content_stats.unique_token_count` | Distinct tokens — indicates vocabulary diversity |
| `keywords.primary_keyword` | Single most representative word with cosine similarity score |
| `keywords.secondary_keywords` | Top 10 related keyphrases (2-3 words) ranked by score |
| `entities` | Named entities detected by spaCy with type labels |
| `all_keyword_scores` | Complete ranked list of all extracted keywords |

### Entity Labels

| Label | Meaning | Example |
|---|---|---|
| `PERSON` | People's names | "Elon Musk" |
| `ORG` | Organizations, companies | "Google", "NASA" |
| `GPE` | Countries, cities, states | "United States", "Tokyo" |
| `DATE` | Dates and periods | "January 2024", "the 2020s" |
| `MONEY` | Monetary values | "$1 billion" |
| `CARDINAL` | Numerals | "three", "100" |
| `WORK_OF_ART` | Titles of books, songs | "The Great Gatsby" |
| `NORP` | Nationalities, groups | "American", "Jewish" |

---

## How the Pipeline Works (Step by Step)

```
URL Input
    │
    ▼
┌─────────────────────────────────┐
│  1. SCRAPER (scraper.py)        │
│  - Download HTML via requests   │
│  - Parse with BeautifulSoup     │
│  - Extract metadata, headings   │
│  - Remove boilerplate content   │
│  - Return clean body text       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  2. spaCy (preprocessor.py)     │
│  - Tokenize body text           │
│  - Lemmatize tokens             │
│  - Remove stop words            │
│  - Detect named entities (NER)  │
│  - Return tokens + entities     │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  3. KeyBERT (keyword_extractor) │
│  - Build focused text from      │
│    title (3x) + description     │
│    (2x) + headings + body intro │
│  - Extract single-word keywords │
│  - Extract 2-3 word keyphrases  │
│  - Apply MMR for diversity      │
│  - Rank by cosine similarity    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  4. JSON Output (main.py)       │
│  - Assemble metadata + stats    │
│  - Combine keywords + entities  │
│  - Pretty-print structured JSON │
└─────────────────────────────────┘
```

---

## Limitations

- **JavaScript-rendered content** — Pages that load content via JS (SPAs, infinite scroll) will return incomplete body text. Only static HTML is scraped.
- **Paywalled content** — Articles behind paywalls will only return the preview/snippet.
- **Language** — The spaCy model and KeyBERT embeddings are English-only. For other languages, swap `en_core_web_sm` for the appropriate spaCy model and use `paraphrase-multilingual-MiniLM-L12-v2` as the KeyBERT model.
- **Very short pages** — Landing pages with minimal text may produce low-confidence keywords.
- **Dynamic class names** — Boilerplate removal uses common CSS class names (.nav, .footer, etc.). Pages with unusual class naming may retain some boilerplate text.

---

## Customization

### Use a larger spaCy model for better NER

```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
```

Then change in `preprocessor.py`:
```python
_nlp = spacy.load("en_core_web_lg")
```

### Use a multilingual KeyBERT model

In `keyword_extractor.py`, change the model parameter:
```python
_kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
```

### Adjust keyword quantity

In `keyword_extractor.py`, modify `top_n` values:
- Higher `top_n` = more candidates extracted
- Lower `diversity` = keywords are more similar to each other (higher scores)
- Higher `diversity` = keywords cover more varied subtopics (lower scores)

### Increase body content window

In `keyword_extractor.py`, change the truncation limit in `_prepare_text_for_keywords`:
```python
parts.append(body[:10000])  # was 5000
```

---

## License

MIT
