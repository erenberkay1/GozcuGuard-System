"""Credit text and a small integrity check.

The credit is stored in pieces and verified against a SHA-256 digest. It is checked in several places
(window title, footer, welcome dialog, pipeline start). This raises the bar for casually removing the
author credit, but it cannot stop a determined person who edits the code. The legal protection is the
MIT license, which requires the copyright and attribution notice to be kept in all copies.
"""
import hashlib

_PARTS = ("Created by ", "Eren Berkay Gün")
_DIGEST = "3b389cb54090825d8004ce5ca47cbf9c415ef85029cc4991bcee8145ca48edc8"


def credit_text() -> str:
    return "".join(_PARTS)


def credit_ok() -> bool:
    try:
        return hashlib.sha256(credit_text().encode("utf-8")).hexdigest() == _DIGEST
    except Exception:
        return False


def verified_credit() -> str:
    """Credit text for display; shows a warning instead if the credit was tampered with."""
    if credit_ok():
        return credit_text()
    return "MODIFIED COPY - original by Eren Berkay Gün"
