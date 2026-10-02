from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from memory.importers.base import ConversationImporter

_MAX_INPUT_BYTES = 20_000_000
_ROLES = {"user", "assistant"}


class PiSessionJsonlImporter(ConversationImporter):
    """Parse Pi-family session-tree JSONL transcripts.

    Covers the shared session lineage of Pi (``pi``), Oh My Pi, and OpenClaw
    archive artifacts: a versioned ``session`` header followed by append-only
    entries linked through ``id``/``parentId``. The transcript is a tree, not a
    linear chat: rewind and fork operations add branches, and the active
    conversation is the ancestry of the last message entry. The importer walks
    that leaf-to-root path and reverses it, so abandoned branches are excluded.

    Legacy pre-tree files whose entries carry no ids are imported in physical
    order. If the active ancestry reaches a ``parentId`` that is absent from
    the file (truncated or hand-edited transcripts), the reachable prefix is
    imported and flagged as ``partial_ancestry`` in metadata. A ``parentId``
    cycle is rejected as corrupt.

    Only user and assistant text is imported. System prompts, thinking blocks,
    tool calls and results, custom records, credential pins, compaction bodies,
    and usage or provider-private metadata are skipped.
    """

    name = "pi-session-jsonl"

    def import_text(
        self,
        text: str,
        *,
        source: str | None = None,
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            raise ValueError("Pi session input must be text")
        if len(text.encode("utf-8")) > _MAX_INPUT_BYTES:
            raise ValueError("Pi session input exceeds 20000000 bytes")

        entries: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Pi session JSONL line {line_number} must be valid JSON"
                ) from exc
            if not isinstance(entry, dict):
                raise ValueError(f"Pi session JSONL line {line_number} must be an object")
            entries.append(entry)
        if not entries:
            raise ValueError("Pi session JSONL contains no entries")

        header, title_slot = _split_header(entries)
        if header is None:
            raise ValueError("Pi session JSONL has no session header")

        variant = _detect_variant(entries)
        message_entries = [entry for entry in entries if entry.get("type") == "message"]
        if not message_entries:
            raise ValueError("Pi session JSONL contains no messages")

        active_entries, partial = _active_branch(entries, message_entries)

        messages: list[dict[str, str]] = []
        message_times: list[datetime] = []
        model = _branch_model(active_entries)
        for entry in active_entries:
            if entry.get("type") != "message":
                continue
            raw_message = entry.get("message")
            if not isinstance(raw_message, dict):
                continue
            role = raw_message.get("role")
            if role not in _ROLES:
                continue
            content = _message_text(raw_message.get("content"))
            if content is None:
                continue
            messages.append({"role": role, "text": content})
            created = _entry_time(entry.get("timestamp"), field="entry timestamp")
            if created is not None:
                message_times.append(created)
            if model is None and role == "assistant":
                model = _model_name(raw_message)

        if not messages:
            raise ValueError("Pi session JSONL contains no conversational text")

        resolved_title = title or _latest_title(entries, header, title_slot)
        metadata: dict[str, Any] = {
            "importer": self.name,
            "platform": "cli",
            "ingestion_method": "json-import",
            "variant": variant,
        }
        if partial:
            metadata["partial_ancestry"] = True
        session_id = _nonempty_string(header.get("id"))
        if session_id:
            metadata["source_session_id"] = session_id
        if resolved_title:
            metadata["title"] = resolved_title
        directory = _nonempty_string(header.get("cwd"))
        if directory:
            metadata["directory"] = directory
        if model:
            metadata["model"] = model

        payload: dict[str, Any] = {
            "source": source or "pi",
            "messages": messages,
            "metadata": metadata,
        }
        if resolved_title:
            payload["title"] = resolved_title

        session_time = _entry_time(header.get("timestamp"), field="session timestamp")
        timestamp = session_time or (min(message_times) if message_times else None)
        if timestamp is not None:
            payload["timestamp"] = timestamp.isoformat()
        return [payload]


def _split_header(
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the session header and the leading fixed-width title slot, if any."""
    title_slot: str | None = None
    for entry in entries:
        entry_type = entry.get("type")
        if entry_type == "title" and title_slot is None:
            title_slot = _nonempty_string(entry.get("title"))
            continue
        if entry_type == "session":
            return entry, title_slot
        break
    return None, title_slot


def _detect_variant(entries: list[dict[str, Any]]) -> str:
    """Detect the producing tool from lineage-specific entry markers."""
    for entry in entries:
        if entry.get("type") != "custom":
            continue
        custom_type = entry.get("customType")
        if isinstance(custom_type, str) and "openclaw" in custom_type.lower():
            return "openclaw"
    for entry in entries:
        if entry.get("type") in {"credential_pin", "title_change", "title"}:
            return "oh-my-pi"
    return "pi"


def _active_branch(
    entries: list[dict[str, Any]],
    message_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Return root-to-leaf entries of the active branch and whether it is partial.

    Legacy entries without ids fall back to physical order. Walking stops at a
    missing parent id (marked partial) and rejects a ``parentId`` cycle.
    """
    leaf = message_entries[-1]
    leaf_id = leaf.get("id")
    if not isinstance(leaf_id, str):
        return list(entries), False

    by_id = {
        entry["id"]: entry
        for entry in entries
        if isinstance(entry.get("id"), str)
    }
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: dict[str, Any] | None = leaf
    partial = False
    while current is not None:
        entry_id = current.get("id")
        if not isinstance(entry_id, str) or entry_id in seen:
            raise ValueError("Pi session JSONL contains a parentId cycle")
        seen.add(entry_id)
        chain.append(current)
        parent_id = current.get("parentId")
        if parent_id is None:
            break
        if not isinstance(parent_id, str) or parent_id not in by_id:
            partial = True
            break
        current = by_id[parent_id]
    chain.reverse()
    return chain, partial


def _branch_model(active_entries: list[dict[str, Any]]) -> str | None:
    """Return the model of the latest model_change on the active branch."""
    for entry in reversed(active_entries):
        if entry.get("type") != "model_change":
            continue
        model = _model_name(entry.get("modelId")) or _model_name(entry.get("model"))
        if model is not None:
            return model
    return None


def _latest_title(
    entries: list[dict[str, Any]],
    header: dict[str, Any],
    title_slot: str | None,
) -> str | None:
    for entry in reversed(entries):
        if entry.get("type") == "title_change":
            title = _nonempty_string(entry.get("title"))
            if title:
                return title
    return title_slot or _nonempty_string(header.get("title"))


def _message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        if text.strip():
            text_parts.append(text)
    if not text_parts:
        return None
    return "\n\n".join(text_parts)


def _entry_time(value: Any, *, field: str) -> datetime | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(value / 1000, UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _model_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, dict):
        return None
    model = _nonempty_string(value.get("model")) or _nonempty_string(value.get("modelId"))
    if model is None:
        return None
    provider = _nonempty_string(value.get("provider"))
    return f"{provider}/{model}" if provider else model


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
