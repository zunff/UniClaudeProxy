import threading
import logging
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger("anyclaude")


class _ConfigReloadHandler(FileSystemEventHandler):
    """Handles config file modification events and triggers reload callback.

    Attributes:
        watched_names: set[str] - Filenames that should trigger reload.
        callback: callable - Function to invoke on config change.
    """

    def __init__(self, watched_names: set[str], callback):
        """Initialize the handler.

        Args:
            watched_names: set[str] - Filenames to watch (e.g. config.json).
            callback: callable - Reload callback function.
        """
        self.watched_names = watched_names
        self.callback = callback
        self._lock = threading.Lock()

    def on_modified(self, event: FileModifiedEvent):
        """Handle file modification events, triggering reload for config changes.

        Args:
            event: FileModifiedEvent - The filesystem event.
        """
        if event.is_directory:
            return

        event_path = Path(event.src_path).resolve()
        if event_path.name in self.watched_names:
            with self._lock:
                logger.info("Config file changed: %s", event_path.name)
                try:
                    self.callback()
                    logger.info("Config reloaded successfully")
                except Exception as e:
                    logger.error("Failed to reload config: %s", e)


class ConfigWatcher:
    """Watches config JSON files for changes and triggers hot reload.

    Attributes:
        paths: list[Path] - Resolved paths to watch.
        callback: callable - Function to invoke on config change.
        observer: Observer | None - The watchdog observer instance.
    """

    def __init__(self, config_path: str | list[str], callback):
        """Initialize the watcher.

        Args:
            config_path: str | list[str] - Path(s) to watch.
            callback: callable - Function to call when config changes.
        """
        raw = [config_path] if isinstance(config_path, str) else list(config_path)
        self.paths = [Path(p).resolve() for p in raw]
        self.callback = callback
        self.observer = None

    def start(self):
        """Start watching the config file(s) for modifications."""
        names = {p.name for p in self.paths}
        handler = _ConfigReloadHandler(names, self.callback)
        self.observer = Observer()
        parents = {str(p.parent) for p in self.paths}
        for parent in parents:
            self.observer.schedule(handler, parent, recursive=False)
        self.observer.start()
        for p in self.paths:
            logger.info("Watching config: %s", p)

    def stop(self):
        """Stop watching the config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Config watcher stopped")
