"""
Transcript sanitisation on ingest.

Some models echo the speaker tag from the history format back into their own
output — Llama reliably produces `[critic]: [critic]: ...`. That artifact would
show up in judged transcripts and in any n-gram redundancy metric, so it gets
stripped where the turn enters the history rather than prompted away.

Pure functions only: these are exactly the "silent bug corrupts published
results" surfaces CLAUDE.md calls out, so they're unit-tested.
"""

from __future__ import annotations

import re

# A leading `[name]:` tag, as produced by _build_history's speaker prefixes.
_SPEAKER_TAG = re.compile(r"^\s*\[(?P<name>[^\]\n]{1,64})\]\s*:\s*")


def strip_speaker_echo(speaker: str, text: str) -> str:
    """
    Remove `[speaker]:` prefixes the speaker echoed back at the start of its turn.

    Only the speaker's *own* tag is stripped, and only from the front. Another
    agent's tag mid-text is meaningful (agents address each other by name), so
    removing it would destroy content.
    """
    remainder = text
    stripped = False
    while (match := _SPEAKER_TAG.match(remainder)):
        if match.group("name").strip().casefold() != speaker.strip().casefold():
            break
        remainder = remainder[match.end():]
        stripped = True
    return remainder.lstrip() if stripped else text
