"""Write plain article text from the archived Wang et al. HTML page."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / "0_reach_topology" / "data" / "raw" / "point_sources" / "wastewater" / "wang_wwtp_2006_2019" / "metadata" / "s41597-022-01439-7.html"
OUT = ROOT / "5_Test" / "20260817_9" / "work" / "wang_article_text.txt"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def main() -> None:
    extractor = TextExtractor()
    extractor.feed(SOURCE.read_text(encoding="utf-8", errors="replace"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(extractor.parts) + "\n", encoding="utf-8")
    print(f"fragments={len(extractor.parts)} output={OUT}")


if __name__ == "__main__":
    main()
