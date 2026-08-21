"""
Turn a run record JSON into a readable HTML page for skimming a conversation:
each turn's text in the middle, the selector's "why" for that turn (bids,
consensus votes, who got addressed, the facilitator's call) as a row of chips
underneath, then the next turn. Self-contained output — no external CSS/JS,
just open the file in a browser.

    uv run render_run.py runs/cheap_test/5d6b4d23630c.json
    uv run render_run.py runs/cheap_test/5d6b4d23630c.json -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

BG = "#EDE8D0"
BLUE = "#135D82"
NAVY = "#023047"
AMBER = "#FFB703"
ORANGE = "#FB8500"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render a run record JSON as a readable HTML page.")
    parser.add_argument("run_json", help="path to a runs/<run_id>.json file")
    parser.add_argument("-o", "--output", default=None, help="output path (default: same name, .html)")
    return parser.parse_args(argv)


# --- minimal markdown, tailored to this repo's story-template shape ---

def _md_lite(text: str, roster: list[str]) -> str:
    text = html.escape(text)
    lines = text.split("\n")
    out: list[str] = []
    para: list[str] = []
    in_list = False

    def flush_para():
        if para:
            out.append("<p>" + " ".join(para) + "</p>")
            para.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            flush_para(); close_list()
            out.append(f"<h4>{stripped[4:]}</h4>")
        elif stripped.startswith("## "):
            flush_para(); close_list()
            out.append(f"<h3>{stripped[3:]}</h3>")
        elif stripped.startswith("- "):
            flush_para()
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{stripped[2:]}</li>")
        elif stripped in ("", "---"):
            flush_para(); close_list()
            if stripped == "---":
                out.append("<hr>")
        else:
            close_list()
            para.append(stripped)
    flush_para(); close_list()

    joined = "\n".join(out)
    joined = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", joined)
    joined = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", joined)
    for name in roster:
        joined = re.sub(
            rf"(\[{re.escape(name)}\]|@{re.escape(name)})",
            rf'<span class="mention">\1</span>', joined,
        )
    return joined


# --- selector "blob" rendering ---

def _chip(title: str, subtitle_html: str, reason: str, accent: str, highlight: bool = False) -> str:
    # subtitle_html is pre-built HTML (e.g. the level meter's <span> dots) or a
    # hardcoded/roster-name-only string — never raw LLM text, so it's trusted
    # as-is; title/reason do carry LLM-generated free text and stay escaped.
    cls = "chip chip-win" if highlight else "chip"
    return (
        f'<div class="{cls}" style="--accent:{accent}">'
        f'<div class="chip-title">{html.escape(title)}</div>'
        f'<div class="chip-subtitle">{subtitle_html}</div>'
        f'<div class="chip-reason">{html.escape(reason)}</div>'
        f"</div>"
    )


def _meter(level: int | None) -> str:
    n = level if isinstance(level, int) else 0
    dots = "".join(f'<span class="dot{" filled" if i < n else ""}"></span>' for i in range(5))
    return f'<span class="meter">{dots}</span>'


def _section(label: str, inner: str) -> str:
    return f'<div class="blob-section"><div class="blob-label">{html.escape(label)}</div><div class="chip-row">{inner}</div></div>'


def _render_bids(sel: dict) -> str:
    bids = sel.get("bids")
    if not bids:
        return ""
    winner = sel.get("winner")
    chips = []
    for b in bids:
        agent = b.get("agent", "?")
        is_winner = agent == winner
        subtitle = _meter(b.get("level")) + (" WINNER" if is_winner else "")
        chips.append(_chip(agent, subtitle, b.get("reason", ""), ORANGE if is_winner else BLUE, is_winner))
    label = f"Bids — winner: {winner}" if winner else "Bids"
    return _section(label, "".join(chips))


def _render_votes(sel: dict) -> str:
    votes = sel.get("consensus_votes")
    if not votes:
        return ""
    chips = [
        _chip(v.get("agent", "?"), "STOP" if v.get("stop") else "continue", v.get("reason", ""),
              AMBER if v.get("stop") else BLUE, bool(v.get("stop")))
        for v in votes
    ]
    stopped = sel.get("consensus_stop")
    label = "Consensus check — MEETING ENDS" if stopped else "Consensus check"
    return _section(label, "".join(chips))


def _render_obligation(sel: dict) -> str:
    ob = sel.get("obligation_check")
    if not ob:
        return ""
    addressed = ob.get("addressed")
    if addressed:
        title, subtitle = ob.get("speaker", "?"), f"addressed {html.escape(addressed)} directly"
    else:
        title, subtitle = ob.get("speaker", "?"), "addressed no one — falls back to bidding"
    return _section("Obligation check", _chip(title, subtitle, ob.get("reason", ""), ORANGE if addressed else BLUE, bool(addressed)))


def _render_facilitator(sel: dict) -> str:
    fd = sel.get("facilitator_decision")
    if not fd:
        return ""
    chair, nxt = fd.get("chair", "?"), fd.get("next", "?")
    subtitle = "picks itself (SELF)" if chair == nxt else f"delegates to {html.escape(nxt)}"
    return _section(f"Facilitator ({chair})", _chip(chair, subtitle, fd.get("reason", ""), ORANGE))


def _render_blob(selector: dict | None) -> str:
    if not selector:
        return ""
    sections = [
        _render_obligation(selector),
        _render_facilitator(selector),
        _render_bids(selector),
        _render_votes(selector),
    ]
    sections = [s for s in sections if s]
    if not sections:
        return ""
    return f'<div class="blob">{"".join(sections)}</div>'


# --- page assembly ---

def _selection_summary(selector: dict | None, speaker: str) -> str:
    """
    One-line "why this speaker" for the turn header — the detailed blob further
    down explains everyone's reasoning, but it sits below the speech content,
    which can be thousands of words long. Without a short answer right at the
    top, "why did X speak next" is easy to miss entirely.
    """
    if not selector:
        return ""
    ob = selector.get("obligation_check")
    if ob and ob.get("addressed") == speaker:
        return f"addressed by {ob.get('speaker', '?')}"
    fd = selector.get("facilitator_decision")
    if fd and fd.get("next") == speaker:
        return "picked itself" if fd.get("chair") == speaker else f"delegated by {fd.get('chair', '?')}"
    bids = selector.get("bids")
    if bids:
        level = next((b.get("level") for b in bids if b.get("agent") == speaker), None)
        if level is not None:
            return f"won bid (level {level})"
    return ""


def _turn_html(turn: dict, roster: list[str]) -> str:
    speaker = turn.get("speaker", "?")
    body = _md_lite(turn.get("content", ""), roster)
    selector = turn.get("selector")
    blob = _render_blob(selector)
    summary = _selection_summary(selector, speaker)
    summary_html = f'<span class="why">{html.escape(summary)}</span>' if summary else ""
    return (
        f'<div class="turn">'
        f'<div class="turn-header"><span class="speaker">{html.escape(speaker)}</span>'
        f"{summary_html}"
        f'<span class="turn-index">turn {turn.get("index", "?")}</span></div>'
        f"{blob}"
        f'<div class="turn-body">{body}</div>'
        f"</div>"
    )


def _outcome_html(consensus: dict | None) -> str:
    if not consensus or consensus.get("votes") is None:
        return ""
    label = "Meeting ended by consensus" if consensus.get("stopped") else "Ran to turn budget — no unanimity"
    chips = "".join(
        _chip(v.get("agent", "?"), "STOP" if v.get("stop") else "continue", v.get("reason", ""),
              AMBER if v.get("stop") else BLUE, bool(v.get("stop")))
        for v in consensus["votes"]
    )
    return f'<div class="outcome"><h2>{html.escape(label)}</h2><div class="chip-row">{chips}</div></div>'


def _summary_html(data: dict, roster: list[str]) -> str:
    summary = data.get("summary")
    if not summary:
        return ""
    return f'<div class="summary"><h2>Final deliverable</h2><div class="turn-body">{_md_lite(summary, roster)}</div></div>'


def _header_html(data: dict) -> str:
    config = data.get("config", {})
    totals = (data.get("totals") or {}).get("all", {})
    strategy = (config.get("strategy") or {}).get("name", "?")
    cost = totals.get("cost_usd")
    cost_str = f"${cost:.4f}" if cost is not None else "n/a"
    if not totals.get("cost_complete", True):
        cost_str += " (partial)"
    badges = [
        ("config", config.get("name", "?")), ("strategy", strategy),
        ("turns", str((data.get("totals") or {}).get("turns", len(data.get("turns", []))))),
        ("cost", cost_str), ("run_id", data.get("run_id", "?")),
    ]
    badge_html = "".join(f'<span class="badge"><b>{k}</b> {html.escape(str(v))}</span>' for k, v in badges)
    errors = data.get("errors") or []
    error_html = ""
    if errors:
        items = "".join(f"<li>{html.escape(e)}</li>" for e in errors)
        error_html = f'<div class="errors">⚠ <ul>{items}</ul></div>'
    return f'<header><h1>Refinement meeting transcript</h1><div class="badges">{badge_html}</div>{error_html}</header>'


CSS = f"""
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 2rem 1rem 4rem; background: {BG};
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  color: {NAVY};
}}
header {{
  max-width: 780px; margin: 0 auto 2.5rem; background: {NAVY}; color: {BG};
  padding: 1.5rem 1.75rem; border-radius: 14px;
}}
header h1 {{ margin: 0 0 0.75rem; font-size: 1.4rem; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
.badge {{
  background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.25);
  border-radius: 999px; padding: 0.25rem 0.75rem; font-size: 0.85rem;
}}
.badge b {{ color: {AMBER}; margin-right: 0.35em; }}
.errors {{ margin-top: 0.75rem; color: {AMBER}; font-size: 0.9rem; }}
.errors ul {{ margin: 0.25rem 0 0; padding-left: 1.5rem; }}

.turn, .outcome, .summary {{
  max-width: 780px; margin: 0 auto 1.25rem; background: #fff;
  border-radius: 14px; box-shadow: 0 2px 10px rgba(2,48,71,0.08); overflow: hidden;
}}
.turn-header {{
  display: flex; align-items: center; gap: 0.6rem;
  background: {BLUE}; color: #fff; padding: 0.6rem 1.25rem;
}}
.speaker {{ font-weight: 700; letter-spacing: 0.02em; }}
.why {{
  font-size: 0.78rem; background: {ORANGE}; color: {NAVY}; font-weight: 700;
  padding: 0.15rem 0.55rem; border-radius: 999px;
}}
.turn-index {{ font-size: 0.8rem; opacity: 0.8; margin-left: auto; }}
.turn-body {{ padding: 1.1rem 1.25rem 0.3rem; line-height: 1.55; font-size: 0.96rem; }}
.turn-body h3 {{ color: {NAVY}; margin: 1rem 0 0.4rem; font-size: 1.05rem; }}
.turn-body h4 {{ color: {BLUE}; margin: 0.8rem 0 0.3rem; font-size: 0.95rem; }}
.turn-body p {{ margin: 0 0 0.8rem; }}
.turn-body ul {{ margin: 0 0 0.8rem; padding-left: 1.3rem; }}
.turn-body hr {{ border: none; border-top: 1px solid {BG}; margin: 1rem 0; }}
.mention {{ background: {AMBER}33; color: {NAVY}; padding: 0 0.3em; border-radius: 4px; font-weight: 600; }}

.blob {{ background: {BG}55; padding: 0.9rem 1.25rem 1.1rem; border-bottom: 1px dashed {BLUE}55; }}
.blob-section + .blob-section {{ margin-top: 0.7rem; }}
.blob-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: {BLUE}; margin-bottom: 0.4rem; font-weight: 700; }}
.chip-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
.chip {{
  --accent: {BLUE}; flex: 1 1 200px; max-width: 260px; background: #fff;
  border: 1.5px solid var(--accent); border-left: 5px solid var(--accent);
  border-radius: 10px; padding: 0.5rem 0.7rem; font-size: 0.82rem;
}}
.chip-win {{ background: {ORANGE}14; }}
.chip-title {{ font-weight: 700; color: {NAVY}; }}
.chip-subtitle {{ color: var(--accent); font-weight: 600; font-size: 0.78rem; margin: 0.1rem 0 0.3rem; }}
.chip-reason {{ color: #333; line-height: 1.35; }}
.meter {{ display: inline-flex; gap: 2px; vertical-align: middle; margin-right: 0.4em; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; background: {BLUE}33; display: inline-block; }}
.dot.filled {{ background: {ORANGE}; }}

.outcome, .summary {{ border: 2px solid {AMBER}; }}
.outcome h2, .summary h2 {{ margin: 0; padding: 0.9rem 1.25rem; background: {AMBER}; color: {NAVY}; font-size: 1.05rem; }}
.outcome .chip-row {{ padding: 1rem 1.25rem; }}
"""


def render(data: dict) -> str:
    roster = list((data.get("config") or {}).get("roster") or [])
    turns_html = "".join(_turn_html(t, roster) for t in data.get("turns", []))
    outcome = _outcome_html(data.get("consensus"))
    summary = _summary_html(data, roster)
    title = html.escape(f"{(data.get('config') or {}).get('name', 'run')} — {data.get('run_id', '')}")
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title><style>{CSS}</style></head><body>"
        f"{_header_html(data)}{turns_html}{outcome}{summary}"
        f"</body></html>"
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    src = Path(args.run_json)
    data = json.loads(src.read_text())
    out = Path(args.output) if args.output else src.with_suffix(".html")
    out.write_text(render(data))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
