"""Markdown-memory backend -- git-diffable, human-readable facts + thread logs."""

from .markdown_memory_manager import MarkdownMemoryManager

#: Discovered by the factory's ``_scan_backends`` drop-in mechanism under the
#: folder name ``markdown_memory``.
MANAGER_CLASS = MarkdownMemoryManager
