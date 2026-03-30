"""
Thin wrappers around JSON file I/O using the standard library `json` module.

Public API:
    load_json(path: str) -> dict       — reads and parses a JSON file
    save_json(data: dict, path: str)   — serialises a dict and writes it to disk
"""
