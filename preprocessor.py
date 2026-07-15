"""
Text preprocessing module using spaCy.
Handles tokenization, lemmatization, stop words removal, and Named Entity Recognition (NER).
"""

import spacy


# Load the spaCy English model with all pipeline components enabled.
# en_core_web_sm includes: tagger, parser, ner, lemmatizer
_nlp = None


def _get_nlp():
    """Lazy-load the spaCy model to avoid reloading on every call."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def preprocess_text(text: str) -> dict:
    """
    Preprocess text using spaCy:
      - Tokenization (automatic in spaCy pipeline)
      - Lemmatization
      - Stop word removal
      - Named Entity Recognition (NER)

    Returns a dict with:
        - tokens: list of cleaned, lemmatized tokens (lowercase, alpha only, no stop words)
        - clean_text: rejoined clean tokens as a string
        - entities: list of detected named entities with text, label, start, end
        - pos_tags: list of (token, POS tag) tuples for content words
    """
    nlp = _get_nlp()
    doc = nlp(text)

    # --- Tokens: lemmatized, lowercase, alpha-only, no stop words ---
    tokens = []
    pos_tags = []
    for token in doc:
        # Skip stop words, punctuation, whitespace, and non-alpha tokens
        if token.is_stop or token.is_punct or token.is_space or not token.is_alpha:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma:  # Skip empty lemmas
            tokens.append(lemma)
            pos_tags.append((lemma, token.pos_))

    # --- Named Entities ---
    entities = []
    seen = set()  # Deduplicate entities
    for ent in doc.ents:
        ent_text = ent.text.strip()
        if ent_text and ent_text.lower() not in seen:
            seen.add(ent_text.lower())
            entities.append({
                "text": ent_text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_) or ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

    # --- Reconstruct clean text from lemmatized tokens ---
    clean_text = " ".join(tokens)

    return {
        "tokens": tokens,
        "clean_text": clean_text,
        "entities": entities,
        "pos_tags": pos_tags,
    }



