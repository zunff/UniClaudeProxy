"""Config hot reload: an atomic write looks like a rename, not a modification."""

import os
import unittest

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileMovedEvent

from app.watcher import _ConfigReloadHandler

CONFIG = os.path.abspath("config.json")
TEMP = os.path.abspath("config.json.tmp")
OTHER = os.path.abspath("prices.json")


class ConfigReloadHandlerTests(unittest.TestCase):
    def _handler(self, names=("config.json",)):
        calls = []
        handler = _ConfigReloadHandler(set(names), lambda: calls.append(1))
        return handler, calls

    def test_in_place_save_reloads(self):
        handler, calls = self._handler()
        handler.on_modified(FileModifiedEvent(CONFIG))
        self.assertEqual(len(calls), 1)

    def test_atomic_write_rename_reloads(self):
        # app.config.atomic_write_json() writes a sibling temp file, then renames
        # it over config.json — which the OS reports as a move, not a modify.
        handler, calls = self._handler()
        handler.on_moved(FileMovedEvent(TEMP, CONFIG))
        self.assertEqual(len(calls), 1)

    def test_atomic_write_create_reloads(self):
        handler, calls = self._handler()
        handler.on_created(FileCreatedEvent(CONFIG))
        self.assertEqual(len(calls), 1)

    def test_unwatched_files_are_ignored(self):
        handler, calls = self._handler()
        handler.on_modified(FileModifiedEvent(OTHER))
        handler.on_moved(FileMovedEvent(OTHER + ".tmp", OTHER))
        handler.on_created(FileCreatedEvent(OTHER))
        self.assertEqual(calls, [])

    def test_events_of_the_temp_file_alone_do_not_reload(self):
        handler, calls = self._handler()
        handler.on_created(FileCreatedEvent(TEMP))
        handler.on_modified(FileModifiedEvent(TEMP))
        self.assertEqual(calls, [])

    def test_a_rename_burst_reloads_once(self):
        handler, calls = self._handler()
        handler.on_moved(FileMovedEvent(TEMP, CONFIG))
        handler.on_created(FileCreatedEvent(CONFIG))
        handler.on_modified(FileModifiedEvent(CONFIG))
        self.assertEqual(len(calls), 1)

    def test_a_later_save_reloads_again(self):
        handler, calls = self._handler()
        handler.on_moved(FileMovedEvent(TEMP, CONFIG))
        handler._last_reload -= handler.DEBOUNCE_SECONDS * 2
        handler.on_moved(FileMovedEvent(TEMP, CONFIG))
        self.assertEqual(len(calls), 2)

    def test_a_failing_reload_does_not_kill_the_watcher(self):
        def boom():
            raise RuntimeError("bad config")

        handler = _ConfigReloadHandler({"config.json"}, boom)
        handler.on_moved(FileMovedEvent(TEMP, CONFIG))  # must not raise
