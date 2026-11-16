"""Unit tests for the markdown-memory backend (backends/markdown_memory/)."""

from __future__ import annotations

from types import SimpleNamespace

from deerflow.agents.memory.backends.markdown_memory import MANAGER_CLASS
from deerflow.agents.memory.backends.markdown_memory.config import MarkdownMemoryConfig
from deerflow.agents.memory.backends.markdown_memory.markdown_memory_manager import MarkdownMemoryManager
from deerflow.agents.memory.manager import MemoryManager, get_memory_manager, reset_memory_manager
from deerflow.config.memory_config import load_memory_config_from_dict


class TestMarkdownMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MarkdownMemoryConfig.from_backend_config({})
        assert cfg.storage_path == ""
        assert cfg.max_facts == 500

    def test_custom_values(self) -> None:
        cfg = MarkdownMemoryConfig.from_backend_config({"storage_path": "/tmp/x", "max_facts": 10})
        assert cfg.storage_path == "/tmp/x"
        assert cfg.max_facts == 10


class TestMarkdownMemoryManagerDiscovery:
    def test_registered_as_a_drop_in_backend(self) -> None:
        """The factory's _scan_backends discovery mechanism finds this folder."""
        assert MANAGER_CLASS is MarkdownMemoryManager
        assert issubclass(MANAGER_CLASS, MemoryManager)

    def test_resolvable_via_manager_class_name(self, tmp_path) -> None:
        reset_memory_manager()
        try:
            load_memory_config_from_dict({"manager_class": "markdown_memory", "backend_config": {"storage_path": str(tmp_path)}})
            manager = get_memory_manager()
            assert isinstance(manager, MarkdownMemoryManager)
        finally:
            reset_memory_manager()


class TestFactRoundTrip:
    def _manager(self, tmp_path, **backend_config) -> MarkdownMemoryManager:
        return MarkdownMemoryManager.from_config({"storage_path": str(tmp_path), **backend_config})

    def test_create_then_get_memory_round_trips(self, tmp_path) -> None:
        m = self._manager(tmp_path)
        doc, fact_id = m.create_fact("User prefers dark mode.", category="preference", confidence=0.8, user_id="u1")
        assert fact_id is not None
        assert doc["facts"][0]["content"] == "User prefers dark mode."

        got = m.get_memory(user_id="u1")
        assert len(got["facts"]) == 1
        assert got["facts"][0]["id"] == fact_id
        assert got["facts"][0]["category"] == "preference"
        assert got["facts"][0]["confidence"] == 0.8

    def test_on_disk_file_is_plain_readable_markdown(self, tmp_path) -> None:
        m = self._manager(tmp_path)
        m.create_fact("User's timezone is UTC+2.", category="context", confidence=0.6, user_id="u1", agent_name="revenue")

        path = tmp_path / "users" / "u1" / "agents" / "revenue" / "memory.md"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        # Human-readable: the raw fact content appears verbatim, not inside a
        # serialized blob (no base64/pickle/json-escaped-string wrapper).
        assert "User's timezone is UTC+2." in text
        assert text.startswith("# Memory facts")
        assert "[context]" in text
        assert "(confidence=0.60)" in text

    def test_update_and_delete_fact(self, tmp_path) -> None:
        m = self._manager(tmp_path)
        _, fact_id = m.create_fact("Draft fact.", user_id="u1")

        updated = m.update_fact(fact_id, content="Corrected fact.", confidence=0.9, user_id="u1")
        assert updated["facts"][0]["content"] == "Corrected fact."
        assert updated["facts"][0]["confidence"] == 0.9

        cleared = m.delete_fact(fact_id, user_id="u1")
        assert cleared["facts"] == []

    def test_second_write_only_adds_one_new_line_not_a_full_rewrite(self, tmp_path) -> None:
        """Diffability check: writing fact #2 changes only one new line in the
        file relative to the state after fact #1 -- existing lines are
        untouched (same text, same order), so a real `git diff` on this file
        would show a single added line, not a rewritten file."""
        m = self._manager(tmp_path)
        m.create_fact("First fact.", user_id="u1")
        path = tmp_path / "users" / "u1" / "agents" / "_global" / "memory.md"
        lines_after_first = path.read_text(encoding="utf-8").splitlines()

        m.create_fact("Second fact.", user_id="u1")
        lines_after_second = path.read_text(encoding="utf-8").splitlines()

        assert lines_after_second[: len(lines_after_first)] == lines_after_first
        assert len(lines_after_second) == len(lines_after_first) + 1

    def test_search_filters_by_substring_and_category(self, tmp_path) -> None:
        m = self._manager(tmp_path)
        m.create_fact("Loves dark mode.", category="preference", user_id="u1")
        m.create_fact("Works in finance.", category="context", user_id="u1")

        results = m.search("dark", user_id="u1")
        assert len(results) == 1
        assert "dark mode" in results[0]["content"]

        results = m.search("", category="context", user_id="u1")
        assert len(results) == 1
        assert results[0]["category"] == "context"

    def test_max_facts_cap_evicts_oldest(self, tmp_path) -> None:
        m = self._manager(tmp_path, max_facts=2)
        m.create_fact("Fact A.", user_id="u1")
        m.create_fact("Fact B.", user_id="u1")
        doc, fact_id_c = m.create_fact("Fact C.", user_id="u1")

        assert len(doc["facts"]) == 2
        contents = {f["content"] for f in doc["facts"]}
        assert contents == {"Fact B.", "Fact C."}
        assert fact_id_c is not None

    def test_clear_memory_empties_the_file(self, tmp_path) -> None:
        m = self._manager(tmp_path)
        m.create_fact("Temporary fact.", user_id="u1")
        cleared = m.clear_memory(user_id="u1")
        assert cleared["facts"] == []
        assert m.get_memory(user_id="u1")["facts"] == []

    def test_get_context_renders_bullets_for_injection(self, tmp_path) -> None:
        m = self._manager(tmp_path)
        m.create_fact("User prefers concise answers.", category="preference", user_id="u1")
        text = m.get_context("u1")
        assert "User prefers concise answers." in text
        assert text.startswith("- [preference]")

    def test_get_context_empty_bucket_returns_empty_string(self, tmp_path) -> None:
        m = self._manager(tmp_path)
        assert m.get_context("nobody") == ""


class TestThreadLog:
    def test_add_appends_a_readable_conversation_block(self, tmp_path) -> None:
        m = MarkdownMemoryManager.from_config({"storage_path": str(tmp_path)})
        human = SimpleNamespace(type="human", content="What's the weather?", additional_kwargs={})
        ai = SimpleNamespace(type="ai", content="It's sunny.", additional_kwargs={})

        m.add("thread-1", [human, ai], user_id="u1")

        path = tmp_path / "users" / "u1" / "agents" / "_global" / "threads" / "thread-1.md"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "What's the weather?" in text
        assert "It's sunny." in text
        assert "**User:**" in text
        assert "**Assistant:**" in text

    def test_add_skips_hidden_messages_by_default(self, tmp_path) -> None:
        m = MarkdownMemoryManager.from_config({"storage_path": str(tmp_path)})
        hidden = SimpleNamespace(type="human", content="secret internal note", additional_kwargs={"hide_from_ui": True})
        ai = SimpleNamespace(type="ai", content="ok", additional_kwargs={})

        m.add("thread-2", [hidden, ai], user_id="u1")

        path = tmp_path / "users" / "u1" / "agents" / "_global" / "threads" / "thread-2.md"
        text = path.read_text(encoding="utf-8")
        assert "secret internal note" not in text

    def test_add_with_no_user_or_assistant_text_writes_nothing(self, tmp_path) -> None:
        m = MarkdownMemoryManager.from_config({"storage_path": str(tmp_path)})
        m.add("thread-3", [], user_id="u1")
        path = tmp_path / "users" / "u1" / "agents" / "_global" / "threads" / "thread-3.md"
        assert not path.exists()

    def test_shutdown_flush_is_a_trivial_success(self, tmp_path) -> None:
        m = MarkdownMemoryManager.from_config({"storage_path": str(tmp_path)})
        assert m.shutdown_flush(timeout=1.0) is True
