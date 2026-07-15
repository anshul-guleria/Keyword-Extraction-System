"""
Web scraping module that extracts metadata and clean body content from a webpage.
Uses BeautifulSoup and requests for HTML parsing and content extraction.
"""

import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse, urljoin


# Tags that typically contain non-content elements to remove
REMOVE_TAGS = [
    "script", "style", "noscript", "iframe", "form", "input", "button",
    "svg", "canvas", "applet", "object", "embed", "map", "head",
]

# CSS classes and IDs commonly associated with boilerplate content
BOILERPLATE_SELECTORS = [
    "nav", "footer", "header",
    '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
    ".nav", ".navbar", ".navigation", ".menu", ".sidebar",
    ".footer", ".header", ".banner",
    ".advertisement", ".ads", ".ad-", ".social", ".share",
    ".cookie", ".popup", ".modal", ".overlay",
    ".widget", ".related", ".comments", ".comment-section",
]


def fetch_page(url: str, timeout: int = 15) -> str:
    """Download the HTML content of a webpage."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_metadata(soup: BeautifulSoup) -> dict:
    """Extract SEO-relevant metadata from the page soup."""
    metadata = {}

    # Page title: prefer og:title, then <title> tag
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        metadata["title"] = og_title["content"].strip()
    elif soup.title and soup.title.string:
        metadata["title"] = soup.title.string.strip()
    else:
        metadata["title"] = ""

    # Meta description: prefer og:description, then meta description
    og_desc = soup.find("meta", property="og:description")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if og_desc and og_desc.get("content"):
        metadata["description"] = og_desc["content"].strip()
    elif meta_desc and meta_desc.get("content"):
        metadata["description"] = meta_desc["content"].strip()
    else:
        metadata["description"] = ""

    # Meta keywords
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    if meta_keywords and meta_keywords.get("content"):
        metadata["keywords"] = meta_keywords["content"].strip()
    else:
        metadata["keywords"] = ""

    # Open Graph and other useful meta tags
    og_type = soup.find("meta", property="og:type")
    metadata["og_type"] = og_type["content"].strip() if og_type and og_type.get("content") else ""

    og_image = soup.find("meta", property="og:image")
    metadata["og_image"] = og_image["content"].strip() if og_image and og_image.get("content") else ""

    # Canonical URL
    canonical = soup.find("link", rel="canonical")
    metadata["canonical_url"] = canonical["href"].strip() if canonical and canonical.get("href") else ""

    # Author
    meta_author = soup.find("meta", attrs={"name": "author"})
    metadata["author"] = meta_author["content"].strip() if meta_author and meta_author.get("content") else ""

    return metadata


def extract_headings(soup: BeautifulSoup) -> dict:
    """Extract all heading tags (h1-h6) from the page."""
    headings = {}
    for level in range(1, 7):
        tag = f"h{level}"
        found = soup.find_all(tag)
        if found:
            headings[tag] = [h.get_text(strip=True) for h in found if h.get_text(strip=True)]
    return headings


def _remove_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove navigation, footer, ads, scripts, and other non-content elements."""

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove script, style, and other non-content tags
    for tag_name in REMOVE_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Remove boilerplate sections by CSS selectors
    for selector in BOILERPLATE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    # Remove elements with hidden display
    for element in soup.find_all(style=True):
        style = element["style"].lower()
        if "display:none" in style.replace(" ", "") or "display: none" in style:
            element.decompose()

    # Remove elements with aria-hidden
    for element in soup.find_all(attrs={"aria-hidden": "true"}):
        element.decompose()

    return soup


def extract_body_content(soup: BeautifulSoup) -> str:
    """Extract clean visible text from the page body, removing boilerplate."""
    # Try to find the main content area first
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("div", class_="content")
        or soup.find("div", id="content")
    )

    # Fall back to body if no main content area found
    if main_content is None:
        main_content = soup.body if soup.body else soup

    # Create a copy to avoid modifying the original
    content_soup = main_content.__copy__() if hasattr(main_content, '__copy__') else BeautifulSoup(str(main_content), "lxml")

    # Remove boilerplate elements from the content area
    content_soup = _remove_boilerplate(content_soup)

    # Get text with spaces between block elements for readability
    for br in content_soup.find_all("br"):
        br.replace_with("\n")

    for p in content_soup.find_all("p"):
        p.insert_after("\n")

    text = content_soup.get_text(separator=" ", strip=True)

    # Clean up excessive whitespace while preserving paragraph breaks
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # Remove empty lines
    return "\n".join(lines)


def scrape_webpage(url: str) -> dict:
    """
    Main scraping function: fetches a URL and returns structured data.

    Returns a dict with:
        - url: The original URL
        - metadata: Page metadata (title, description, keywords, etc.)
        - headings: All heading tags grouped by level
        - body_content: Clean visible text from the page
    """
    html = fetch_page(url)
    soup = BeautifulSoup(html, "lxml")

    metadata = extract_metadata(soup)
    headings = extract_headings(soup)
    body_content = extract_body_content(soup)

    return {
        "url": url,
        "domain": urlparse(url).netloc,
        "metadata": metadata,
        "headings": headings,
        "body_content": body_content,
    }
