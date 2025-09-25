import html
import re
from typing import List


_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+|www\.\S+")
_NON_ASCII = re.compile(r"[^a-zA-Z0-9\s.,!?'-]")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = html.unescape(text)
	text = _HTML_TAG.sub(" ", text)
	text = _URL.sub(" ", text)
	text = _NON_ASCII.sub(" ", text)
	text = _MULTI_SPACE.sub(" ", text).strip().lower()
	return text


def batch_clean_text(texts: List[str]) -> List[str]:
	return [clean_text(t) for t in texts]
