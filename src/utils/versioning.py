"""
Provides the current Git commit SHA to be stamped into model_info.json,
enabling full traceability between a saved model artifact and the exact code
revision that produced it.

Public API:
    get_git_sha() -> str   — returns the short commit SHA via subprocess
"""
