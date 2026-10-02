from __future__ import annotations

from pathlib import Path

import pytest

from memory.importers import (
    ConversationImporter,
    CopilotActivityCsvImporter,
    DeepSeekShareJsonImporter,
    ManualPasteImporter,
    OpenCodeSessionJsonImporter,
    PiSessionJsonlImporter,
    get_importer,
    importer_names,
)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "importers"


def test_manual_paste_importer_returns_unified_payload() -> None:
    importer = ManualPasteImporter()

    payloads = importer.import_text(
        "You: Can you remember this?\ncontinued detail\nCopilot: Yes, I can.",
        source="vscode-copilot",
        title="Memory discussion",
    )

    assert isinstance(importer, ConversationImporter)
    assert payloads == [
        {
            "source": "vscode-copilot",
            "title": "Memory discussion",
            "messages": [
                {"role": "user", "text": "Can you remember this?\ncontinued detail"},
                {"role": "assistant", "text": "Yes, I can."},
            ],
            "metadata": {"importer": "manual"},
        }
    ]


@pytest.mark.parametrize("speaker", ["User", "You", "Human"])
def test_manual_paste_importer_normalizes_user_aliases(speaker: str) -> None:
    payload = ManualPasteImporter().import_text(f"{speaker}: Hello\nAssistant: Hi")[0]

    assert [message["role"] for message in payload["messages"]] == ["user", "assistant"]


@pytest.mark.parametrize("speaker", ["Assistant", "AI", "Bot", "Claude", "Gemini", "ChatGPT"])
def test_manual_paste_importer_normalizes_assistant_aliases(speaker: str) -> None:
    payload = ManualPasteImporter().import_text(f"Human: Hello\n{speaker}: Hi")[0]

    assert [message["role"] for message in payload["messages"]] == ["user", "assistant"]


def test_manual_paste_importer_preserves_multiline_content() -> None:
    payload = ManualPasteImporter().import_text(
        "\n"
        "user: show this snippet\n"
        "```python\n"
        "print('Assistant: still code')\n"
        "```\n"
        "\n"
        "chatgpt:Sure\n"
        "Note: unsupported labels stay in the message body"
    )[0]

    assert payload == {
        "source": "manual-paste",
        "messages": [
            {
                "role": "user",
                "text": (
                    "show this snippet\n"
                    "```python\n"
                    "print('Assistant: still code')\n"
                    "```\n"
                ),
            },
            {
                "role": "assistant",
                "text": "Sure\nNote: unsupported labels stay in the message body",
            },
        ],
        "metadata": {"importer": "manual"},
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("unlabelled text", "first speaker label"),
        ("", "supported speaker labels"),
        ("User:\nAssistant: response", "empty message"),
        ("User:   \nAssistant: response", "empty message"),
    ],
)
def test_manual_paste_importer_rejects_ambiguous_input(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ManualPasteImporter().import_text(text)


def test_manual_paste_importer_rejects_non_text_input() -> None:
    with pytest.raises(ValueError, match="must be text"):
        ManualPasteImporter().import_text(b"User: hello")  # type: ignore[arg-type]


def test_manual_paste_importer_rejects_oversized_input() -> None:
    with pytest.raises(ValueError, match="exceeds 5000000 bytes"):
        ManualPasteImporter().import_text(f"User: {'x' * 5_000_000}")


def test_importer_registry_exposes_manual_importer() -> None:
    assert importer_names() == (
        "copilot-activity-csv",
        "deepseek-share-json",
        "manual",
        "opencode-session-json",
        "pi-session-jsonl",
    )
    assert get_importer("manual").name == "manual"


def test_importer_registry_rejects_unknown_importer() -> None:
    with pytest.raises(
        ValueError,
        match="copilot-activity-csv, deepseek-share-json, manual, opencode-session-json, "
        "pi-session-jsonl",
    ):
        get_importer("unknown")


def test_copilot_activity_csv_fixture_is_grouped_and_chronological() -> None:
    text = (_FIXTURES / "copilot_activity_anonymized.csv").read_text(encoding="utf-8")

    payloads = CopilotActivityCsvImporter().import_text(text)

    assert [payload["title"] for payload in payloads] == [
        "Garden sensor setup",
        "Recipe notes",
    ]
    assert payloads[0]["timestamp"] == "2026-04-12T10:05:00"
    assert [message["role"] for message in payloads[0]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert payloads[1]["messages"][0]["text"].endswith("emoji 🥣.")
    assert "```text\nmix for 30 seconds\n```" in payloads[1]["messages"][1]["text"]
    assert payloads[0]["metadata"] == {
        "importer": "copilot-activity-csv",
        "platform": "web",
        "ingestion_method": "csv-import",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Conversation,Time,Author\nA,2026-01-01T00:00:00,Human", "must have headers"),
        (
            "Conversation,Time,Author,Message\nA,not-a-time,Human,hello",
            "invalid time",
        ),
        (
            "Conversation,Time,Author,Message\nA,2026-01-01T00:00:00,System,hello",
            "unknown author",
        ),
    ],
)
def test_copilot_activity_csv_rejects_invalid_exports(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CopilotActivityCsvImporter().import_text(text)


def test_deepseek_share_json_fixture_flattens_conversational_fragments() -> None:
    text = (_FIXTURES / "deepseek_share_anonymized.json").read_text(encoding="utf-8")

    payload = DeepSeekShareJsonImporter().import_text(text)[0]

    assert payload["source"] == "deepseek"
    assert payload["title"] == "Balcony herb plan"
    assert payload["timestamp"] == "2026-04-12T10:00:00.237000+00:00"
    assert payload["messages"] == [
        {"role": "user", "text": "Suggest herbs for a windy balcony."},
        {
            "role": "assistant",
            "text": "Try rosemary and thyme in weighted pots [1].",
        },
        {"role": "user", "text": "Attachments: balcony-layout.webp"},
        {
            "role": "assistant",
            "text": "Place the heavier pots along the exposed edge.",
        },
    ]
    assert payload["metadata"] == {
        "importer": "deepseek-share-json",
        "platform": "web",
        "ingestion_method": "json-import",
        "model": "default",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "valid JSON"),
        ("{}", "unsupported envelope"),
        ('{"data":{"biz_data":{"messages":[]}}}', "contains no messages"),
    ],
)
def test_deepseek_share_json_rejects_invalid_exports(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeepSeekShareJsonImporter().import_text(text)


def test_opencode_session_json_fixture_maps_text_and_provenance() -> None:
    text = (_FIXTURES / "opencode_session_anonymized.json").read_text(encoding="utf-8")

    payload = OpenCodeSessionJsonImporter().import_text(text)[0]

    assert payload == {
        "source": "opencode",
        "title": "Garden sensor setup",
        "timestamp": "2026-04-12T10:00:00+00:00",
        "messages": [
            {
                "role": "user",
                "text": "Draft a setup checklist for a balcony temperature sensor.",
            },
            {
                "role": "assistant",
                "text": (
                    "1. Mount the sensor in shade.\n2. Record a baseline reading.\n\n"
                    "Keep the enclosure clear of standing water."
                ),
            },
            {"role": "user", "text": "Add a note about battery checks."},
        ],
        "metadata": {
            "importer": "opencode-session-json",
            "platform": "cli",
            "ingestion_method": "json-import",
            "source_session_id": "ses_synthetic_garden",
            "title": "Garden sensor setup",
            "directory": "/workspace/garden-sensor",
            "model": "example-provider/example-model",
        },
    }


def test_opencode_session_json_accepts_overrides_and_message_model() -> None:
    payload = OpenCodeSessionJsonImporter().import_text(
        """{
          "info": {"id": "ses_1", "path": "/workspace/example"},
          "messages": [{
            "info": {
              "role": "assistant",
              "providerID": "provider",
              "modelID": "model",
              "time": {"created": "2026-04-12T10:00:00Z"}
            },
            "parts": [{"type": "text", "text": "Done."}]
          }]
        }""",
        source="custom-opencode",
        title="Override",
    )[0]

    assert payload["source"] == "custom-opencode"
    assert payload["title"] == "Override"
    assert payload["timestamp"] == "2026-04-12T10:00:00+00:00"
    assert payload["metadata"]["title"] == "Override"
    assert payload["metadata"]["directory"] == "/workspace/example"
    assert payload["metadata"]["model"] == "provider/model"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "valid JSON"),
        ("{}", "valid info object"),
        ('{"info": {}, "messages": []}', "contains no messages"),
        (
            '{"info": {}, "messages": [{"info": {"role": "system"}, "parts": []}]}',
            "unknown role",
        ),
        (
            '{"info": {}, "messages": [{"info": {"role": "user"}, "parts": []}]}',
            "no conversational text",
        ),
    ],
)
def test_opencode_session_json_rejects_invalid_exports(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenCodeSessionJsonImporter().import_text(text)


def test_pi_session_jsonl_fixture_imports_active_branch() -> None:
    text = (_FIXTURES / "pi_session_anonymized.jsonl").read_text(encoding="utf-8")

    payload = PiSessionJsonlImporter().import_text(text)[0]

    assert payload == {
        "source": "pi",
        "title": "Sensor checklist",
        "timestamp": "2026-04-12T09:15:00+00:00",
        "messages": [
            {
                "role": "user",
                "text": "Draft a setup checklist for a balcony temperature sensor.",
            },
            {
                "role": "assistant",
                "text": "1. Mount the sensor in shade.\n2. Record a baseline reading.",
            },
            {"role": "user", "text": "Also suggest a shade-friendly mount."},
            {
                "role": "assistant",
                "text": (
                    "Use a bracket under the eave; it keeps direct sun off the housing."
                    "\n\nRecheck the seal after the first rain."
                ),
            },
        ],
        "metadata": {
            "importer": "pi-session-jsonl",
            "platform": "cli",
            "ingestion_method": "json-import",
            "variant": "pi",
            "source_session_id": "0192f0c0-a000-7000-8000-000000000001",
            "title": "Sensor checklist",
            "directory": "/workspace/garden-sensor",
            "model": "example-provider/example-model",
        },
    }


def test_pi_session_jsonl_maps_oh_my_pi_markers() -> None:
    payload = PiSessionJsonlImporter().import_text(
        "\n".join(
            [
                '{"type":"title","title":"Herb plan"}',
                '{"type":"session","version":3,"id":"0192f0c0-b000-7000-8000-000000000002",'
                '"timestamp":"2026-04-12T10:00:00Z","cwd":"/workspace/herb-plan"}',
                '{"type":"message","id":"11","parentId":null,"timestamp":"2026-04-12T10:01:00Z",'
                '"message":{"role":"user","content":[{"type":"text","text":"Suggest herbs for a windy balcony."}]}}',
                '{"type":"credential_pin","id":"12","parentId":"11","timestamp":"2026-04-12T10:01:01Z",'
                '"provider":"example-provider","accountHash":"ab12cd34"}',
                '{"type":"model_change","id":"13","parentId":"12","timestamp":"2026-04-12T10:01:02Z",'
                '"model":"example-provider/herb-model"}',
                '{"type":"message","id":"14","parentId":"13","timestamp":"2026-04-12T10:02:00Z",'
                '"message":{"role":"assistant","content":['
                '{"type":"thinking","thinking":"Weigh pot options."},'
                '{"type":"toolCall","name":"search","arguments":{"query":"windy balcony herbs"}},'
                '{"type":"text","text":"Try rosemary and thyme in weighted pots."}]}}',
                '{"type":"message","id":"15","parentId":"14","timestamp":"2026-04-12T10:02:30Z",'
                '"message":{"role":"toolResult","toolCallId":"t1","toolName":"search",'
                '"content":[{"type":"text","text":"rosemary, thyme, lavender"}]}}',
                '{"type":"title_change","id":"16","parentId":"15","timestamp":"2026-04-12T10:03:00Z",'
                '"title":"Balcony herbs","source":"user"}',
                '{"type":"message","id":"17","parentId":"16","timestamp":"2026-04-12T10:04:00Z",'
                '"message":{"role":"user","content":[{"type":"text","text":"What about wind protection?"}]}}',
                '{"type":"message","id":"18","parentId":"17","timestamp":"2026-04-12T10:05:00Z",'
                '"message":{"role":"assistant","content":[{"type":"text","text":"A low glass screen blocks the gusts."}]}}',
            ]
        )
    )[0]

    assert payload["title"] == "Balcony herbs"
    assert payload["timestamp"] == "2026-04-12T10:00:00+00:00"
    assert payload["messages"] == [
        {"role": "user", "text": "Suggest herbs for a windy balcony."},
        {"role": "assistant", "text": "Try rosemary and thyme in weighted pots."},
        {"role": "user", "text": "What about wind protection?"},
        {"role": "assistant", "text": "A low glass screen blocks the gusts."},
    ]
    assert payload["metadata"]["variant"] == "oh-my-pi"
    assert payload["metadata"]["model"] == "example-provider/herb-model"
    assert payload["metadata"]["title"] == "Balcony herbs"
    assert "partial_ancestry" not in payload["metadata"]


def test_pi_session_jsonl_accepts_legacy_linear_string_content() -> None:
    payload = PiSessionJsonlImporter().import_text(
        "\n".join(
            [
                '{"type":"session","version":1,"id":"legacy-1","timestamp":"2026-04-12T11:00:00Z",'
                '"cwd":"/workspace/legacy"}',
                '{"type":"message","timestamp":"2026-04-12T11:01:00Z",'
                '"message":{"role":"user","content":"Which herbs survive shade?"}}',
                '{"type":"message","timestamp":"2026-04-12T11:02:00Z",'
                '"message":{"role":"assistant","content":"Mint and parsley both tolerate it."}}',
                '{"type":"message","timestamp":"2026-04-12T11:03:00Z",'
                '"message":{"role":"system","content":"Internal prompt text"}}',
                '{"type":"message","timestamp":"2026-04-12T11:04:00Z",'
                '"message":{"role":"user","content":"Thanks."}}',
            ]
        )
    )[0]

    assert payload["messages"] == [
        {"role": "user", "text": "Which herbs survive shade?"},
        {"role": "assistant", "text": "Mint and parsley both tolerate it."},
        {"role": "user", "text": "Thanks."},
    ]
    assert payload["metadata"]["variant"] == "pi"
    assert "model" not in payload["metadata"]


def test_pi_session_jsonl_flags_partial_ancestry_on_missing_parent() -> None:
    payload = PiSessionJsonlImporter().import_text(
        "\n".join(
            [
                '{"type":"session","version":2,"id":"partial-1","timestamp":"2026-04-12T12:00:00Z",'
                '"cwd":"/workspace/partial"}',
                '{"type":"message","id":"21","parentId":null,"timestamp":"2026-04-12T12:01:00Z",'
                '"message":{"role":"user","content":[{"type":"text","text":"Earlier turn."}]}}',
                '{"type":"message","id":"22","parentId":"deadbeef","timestamp":"2026-04-12T12:02:00Z",'
                '"message":{"role":"assistant","content":[{"type":"text","text":"Reachable turn."}]}}',
            ]
        )
    )[0]

    assert [message["text"] for message in payload["messages"]] == ["Reachable turn."]
    assert payload["metadata"]["partial_ancestry"] is True


def test_pi_session_jsonl_rejects_corrupt_cycles() -> None:
    with pytest.raises(ValueError, match="parentId cycle"):
        PiSessionJsonlImporter().import_text(
            "\n".join(
                [
                    '{"type":"session","version":2,"id":"cycle-1","timestamp":"2026-04-12T13:00:00Z",'
                    '"cwd":"/workspace/cycle"}',
                    '{"type":"message","id":"31","parentId":"32","timestamp":"2026-04-12T13:01:00Z",'
                    '"message":{"role":"user","content":[{"type":"text","text":"Loop one."}]}}',
                    '{"type":"message","id":"32","parentId":"31","timestamp":"2026-04-12T13:02:00Z",'
                    '"message":{"role":"assistant","content":[{"type":"text","text":"Loop two."}]}}',
                ]
            )
        )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "valid JSON"),
        ("{}", "session header"),
        (
            '{"type":"session","version":2,"id":"x1","timestamp":"2026-04-12T14:00:00Z"}\n'
            '{"type":"model_change","id":"41","parentId":null,"modelId":"m"}',
            "contains no messages",
        ),
        (
            '{"type":"session","version":2,"id":"x2","timestamp":"2026-04-12T14:00:00Z"}\n'
            '{"type":"message","id":"51","parentId":null,"message":{"role":"toolResult",'
            '"content":[{"type":"text","text":"tool output"}]}}',
            "no conversational text",
        ),
    ],
)
def test_pi_session_jsonl_rejects_invalid_exports(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PiSessionJsonlImporter().import_text(text)
