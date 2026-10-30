# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — every public documentation image has alt text

"""Every image in the public documentation carries non-empty alt text.

The scan covers ``README.md`` and every Markdown page under ``docs/`` except
the private ``docs/internal`` tree, in both the Markdown ``![alt](src)`` and
the HTML ``<img alt="...">`` spellings. Code fences are skipped, because an
example there is not a rendered image.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(")
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
_ALT = re.compile(r"""\balt\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)


def _pages() -> list[Path]:
    """Return the public Markdown pages."""
    docs = ROOT / "docs"
    pages = [p for p in sorted(docs.rglob("*.md")) if (docs / "internal") not in p.parents]
    return [ROOT / "README.md", *pages]


def missing_alt(text: str) -> list[str]:
    """Return every image in ``text`` whose alt text is absent or blank."""
    prose = re.sub(r"^(```|~~~).*?^\1", "", text, flags=re.MULTILINE | re.DOTALL)
    found = [f"![{alt}](" for alt in _MARKDOWN_IMAGE.findall(prose) if not alt.strip()]
    for tag in _HTML_IMAGE.findall(prose):
        match = _ALT.search(tag)
        if match is None or not (match.group(1) or match.group(2) or "").strip():
            found.append(" ".join(tag.split()))
    return found


def test_every_public_documentation_image_has_alt_text() -> None:
    """Refuse a public image without a text alternative."""
    offenders = {
        page.relative_to(ROOT).as_posix(): images
        for page in _pages()
        if (images := missing_alt(page.read_text(encoding="utf-8")))
    }
    assert offenders == {}
    assert len(_pages()) > 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![](a.png)", ["![]("]),
        ("![ ](a.png)", ["![ ]("]),
        ("![Plot of q95](a.png)", []),
        ('<img src="a.png">', ['<img src="a.png">']),
        ('<img src="a.png" alt="">', ['<img src="a.png" alt="">']),
        ("<img\n  src='a.png'\n  alt='Header'>", []),
        ("```markdown\n![](example.png)\n```\n", []),
    ],
)
def test_scanner_finds_images_without_alt_text(text: str, expected: list[str]) -> None:
    """Find blank Markdown and HTML alt text, and ignore code fences."""
    assert missing_alt(text) == expected
