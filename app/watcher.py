import threading
import time
import logging
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger("anyclaude")


class _ConfigReloadHandler(FileSystemEventHandler):
    """Handles config file changes and triggers the reload callback.

    A config file is not only *modified* when it changes: the admin API and
    :func:`app.config.atomic_write_json` write a sibling temp file and rename it
    over the target, which the OS reports as a create/move. Handling only
    ``on_modified`` silently missed every save made that way, so the proxy kept
    running the old config until a restart.

    Attributes:
        watched_names: set[str] - Filenames that should trigger reload.
        callback: callable - Function to invoke on config change.
    """

    #: A rename fires several events; collapse them into one reload.
    DEBOUNCE_SECONDS = 0.3

    def __init__(self, watched_names: set[str], callback):
        """Initialize the handler.

        Args:
            watched_names: set[str] - Filenames to watch (e.g. config.json).
            callback: callable - Reload callback function.
        """
        self.watched_names = watched_names
        self.callback = callback
        self._lock = threading.Lock()
        self._last_reload = 0.0

    def _reload_if_watched(self, path) -> None:
        """Reload when ``path`` is one of the watched files."""
        if not path:
            return
        event_path = Path(path).resolve()
        if event_path.name not in self.watched_names:
            return
        with self._lock:
            now = time.monotonic()
            if now - self._last_reload < self.DEBOUNCE_SECONDS:
                return
            self._last_reload = now
            logger.info("Config file changed: %s", event_path.name)
            try:
                self.callback()
                logger.info("Config reloaded successfully")
            except Exception as e:
                logger.error("Failed to reload config: %s", e)

    def on_modified(self, event) -> None:
        """In-place write (editor save)."""
        if not event.is_directory:
            self._reload_if_watched(event.src_path)

    def on_created(self, event) -> None:
        """Atomic write: the temp file is renamed into place."""
        if not event.is_directory:
            self._reload_if_watched(event.src_path)

    def on_moved(self, event) -> None:
        """The rename half of an atomic write — the target is ``dest_path``."""
        if not event.is_directory:
            self._reload_if_watched(getattr(event, "dest_path", None))


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
