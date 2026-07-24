# Folio — project context for Claude Code

> Project-specific context layered on top of the global standards in `~/.claude/CLAUDE.md`.

## What it does
**Search your own documents the way you search the web — with cited answers, not keyword matches.**

## Stack
python

## Commands
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Conventions
- Conventional commits, one logical change each; secrets never hardcoded; external API calls via a service layer; errors normalized before the client.
- Python: virtualenv always, `requirements.txt` pinned, deterministic logic split from I/O.

