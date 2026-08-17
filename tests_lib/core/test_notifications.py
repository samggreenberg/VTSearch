"""Tests for the user-notification broker (``vtscore.concurrency.notifications``)."""

import logging

import pytest

from vtscore.concurrency.notifications import (
    DEFAULT_LEVEL,
    LEVELS,
    MAX_DETAIL_CHARS,
    MAX_MESSAGE_CHARS,
    Notification,
    NotificationBroker,
    notifications,
    notify,
)


@pytest.fixture
def collected():
    """Subscribe a collector to the process-wide broker for one test."""
    received: list[Notification] = []
    notifications.subscribe(received.append)
    try:
        yield received
    finally:
        notifications.unsubscribe(received.append)


class TestNotificationBroker:
    def test_subscriber_receives_published_notification(self):
        broker = NotificationBroker()
        received: list[Notification] = []
        broker.subscribe(received.append)

        note = Notification(id="note_1", level="info", message="Hello")
        broker.publish(note)

        assert received == [note]

    def test_unsubscribe_stops_delivery(self):
        broker = NotificationBroker()
        received: list[Notification] = []
        broker.subscribe(received.append)
        broker.unsubscribe(received.append)

        broker.publish(Notification(id="note_1", level="info", message="Hello"))
        assert received == []

    def test_unsubscribe_missing_is_noop(self):
        NotificationBroker().unsubscribe(lambda _n: None)  # never subscribed

    def test_subscriber_exception_does_not_break_others(self):
        broker = NotificationBroker()
        seen: list[Notification] = []

        def boom(_n: Notification) -> None:
            raise RuntimeError("subscriber is broken")

        broker.subscribe(boom)
        broker.subscribe(seen.append)

        broker.publish(Notification(id="note_1", level="warning", message="Hi"))
        assert len(seen) == 1

    def test_every_subscriber_receives_each_notification(self):
        broker = NotificationBroker()
        first: list[Notification] = []
        second: list[Notification] = []
        broker.subscribe(first.append)
        broker.subscribe(second.append)

        broker.publish(Notification(id="note_1", level="info", message="Broadcast"))

        assert len(first) == 1 and len(second) == 1

    def test_subscriber_count_and_clear(self):
        broker = NotificationBroker()
        assert broker.subscriber_count() == 0
        broker.subscribe(lambda _n: None)
        broker.subscribe(lambda _n: None)
        assert broker.subscriber_count() == 2
        broker.clear_subscribers()
        assert broker.subscriber_count() == 0


class TestNotify:
    def test_publishes_to_subscribers(self, collected):
        notify("Skipped 3 files", level="warning", detail="a, b, c", source="Server Folder")

        assert len(collected) == 1
        note = collected[0]
        assert note.level == "warning"
        assert note.message == "Skipped 3 files"
        assert note.detail == "a, b, c"
        assert note.source == "Server Folder"

    def test_defaults_to_info(self, collected):
        notify("Just so you know")
        assert collected[0].level == DEFAULT_LEVEL == "info"

    @pytest.mark.parametrize("level", LEVELS)
    def test_every_declared_level_round_trips(self, collected, level):
        notify("Message", level=level)
        assert collected[-1].level == level

    def test_unknown_level_falls_back_instead_of_raising(self, collected):
        note = notify("Message", level="catastrophe")

        assert note.level == DEFAULT_LEVEL
        assert collected[0].level == DEFAULT_LEVEL

    def test_ids_are_unique(self, collected):
        notify("One")
        notify("Two")
        assert collected[0].id != collected[1].id

    def test_blank_message_is_not_broadcast(self, collected):
        note = notify("   ")

        # Returned (callers may inspect it) but never shown: an empty toast is
        # worse than no toast.
        assert note.message == ""
        assert collected == []

    def test_message_is_truncated(self, collected):
        notify("x" * (MAX_MESSAGE_CHARS + 500))

        assert len(collected[0].message) == MAX_MESSAGE_CHARS
        assert collected[0].message.endswith("…")

    def test_detail_is_truncated(self, collected):
        notify("Headline", detail="y" * (MAX_DETAIL_CHARS + 500))

        assert len(collected[0].detail) == MAX_DETAIL_CHARS
        assert collected[0].detail.endswith("…")

    def test_short_message_is_not_truncated(self, collected):
        notify("Short and sweet")
        assert collected[0].message == "Short and sweet"

    def test_empty_detail_and_source_become_none(self, collected):
        notify("Headline", detail="   ", source="  ")

        assert collected[0].detail is None
        assert collected[0].source is None

    def test_broken_subscriber_does_not_propagate_to_caller(self):
        def boom(_n: Notification) -> None:
            raise RuntimeError("subscriber is broken")

        notifications.subscribe(boom)
        try:
            notify("Still fine")  # must not raise
        finally:
            notifications.unsubscribe(boom)

    @pytest.mark.parametrize(
        "level,expected",
        [("info", logging.INFO), ("success", logging.INFO), ("warning", logging.WARNING), ("error", logging.ERROR)],
    )
    def test_logs_at_matching_severity(self, caplog, level, expected):
        with caplog.at_level(logging.INFO, logger="vtscore.concurrency.notifications"):
            notify("Something happened", level=level, source="Plugin")

        records = [r for r in caplog.records if r.name == "vtscore.concurrency.notifications"]
        assert records, "notify() must always log, even with no subscribers"
        assert records[-1].levelno == expected
        assert "Something happened" in records[-1].getMessage()
        assert "Plugin" in records[-1].getMessage()

    def test_logs_even_with_no_subscribers(self, caplog):
        notifications.clear_subscribers()
        with caplog.at_level(logging.INFO, logger="vtscore.concurrency.notifications"):
            notify("Nobody is listening")

        assert any("Nobody is listening" in r.getMessage() for r in caplog.records)

    def test_to_dict_is_json_shaped(self, collected):
        notify("Headline", level="error", detail="Because reasons", source="Exporter")

        payload = collected[0].to_dict()
        assert set(payload) == {"id", "level", "message", "detail", "source", "timestamp"}
        assert payload["level"] == "error"
        assert payload["message"] == "Headline"
        assert payload["detail"] == "Because reasons"
        assert payload["source"] == "Exporter"
        assert isinstance(payload["timestamp"], float)


class TestPluginBaseNotify:
    def test_plugin_notify_attaches_display_name_as_source(self, collected):
        from vtscore.plugins import PluginBase

        class WidgetExporter(PluginBase):
            display_name = "Widget Exporter"
            fields = []

        WidgetExporter().notify("Partial export", level="warning", detail="2 rows dropped")

        assert len(collected) == 1
        assert collected[0].source == "Widget Exporter"
        assert collected[0].level == "warning"
        assert collected[0].detail == "2 rows dropped"

    def test_plugin_notify_defaults_to_info(self, collected):
        from vtscore.plugins import PluginBase

        class QuietImporter(PluginBase):
            display_name = "Quiet Importer"
            fields = []

        QuietImporter().notify("Nothing to report")

        assert collected[0].level == "info"
