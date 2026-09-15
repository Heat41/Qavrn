from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class FileWatcher:
    """
    Memantau satu atau beberapa folder secara otomatis.

    Saat sebuah folder mulai dipantau:
    1. Folder langsung dipindai dan file yang didukung di-index.
    2. Watchdog kemudian memantau perubahan filesystem.
    3. File baru -> otomatis di-index.
    4. File berubah -> otomatis di-index ulang.
    5. File dihapus -> dihapus dari vector store.

    Perubahan file menggunakan debounce agar satu file yang mengalami
    banyak event filesystem tidak diproses berulang kali.
    """

    _DEBOUNCE_SECONDS = 2.0

    def __init__(
            self,
            indexer: Any,
            supported_extensions: List[str],
    ) -> None:
        self._indexer = indexer
        self._exts: Set[str] = {
            e.lower() for e in supported_extensions
        }

        self._observer: Any = None

        # resolved-path -> watchdog Watch handle
        self._watches: Dict[str, Any] = {}

        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def watched_folders(self) -> List[str]:
        return list(self._watches.keys())

    def start(self) -> None:
        """Memulai background filesystem observer."""
        try:
            from watchdog.observers import Observer  # type: ignore
        except ImportError:
            raise ImportError(
                "watchdog diperlukan: pip install watchdog"
            )

        self._observer = Observer()
        self._observer.start()

        logger.info("FileWatcher dimulai.")

    def stop(self) -> None:
        """Menghentikan observer dan timer yang masih tertunda."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()

            self._timers.clear()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        logger.info("FileWatcher dihentikan.")

    def watch(
            self,
            folder: str,
            scan_initial: bool = True,
    ) -> None:
        """
        Mulai memantau folder secara recursive.

        Secara default initial scan tetap dilakukan agar perilaku publik
        lama tidak berubah. Startup API dapat memakai scan_initial=False
        lalu menjalankan scan di background.
        """
        if self._observer is None:
            raise RuntimeError(
                "FileWatcher belum dimulai. Panggil start() terlebih dahulu."
            )

        root = Path(folder).resolve()

        if not root.exists():
            logger.warning(
                "Folder tidak ditemukan: %s",
                root,
            )
            return

        if not root.is_dir():
            logger.warning(
                "Path bukan folder: %s",
                root,
            )
            return

        key = str(root)

        if key in self._watches:
            logger.debug(
                "Folder sudah dipantau: %s",
                key,
            )
            return

        # --------------------------------------------------------------
        # 1. INITIAL SCAN (opsional)
        # --------------------------------------------------------------

        if scan_initial:
            self.scan_initial(root)

        # --------------------------------------------------------------
        # 2. START WATCHDOG
        # --------------------------------------------------------------

        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
        except ImportError:
            raise ImportError(
                "watchdog diperlukan: pip install watchdog"
            )

        watcher = self

        class _Handler(FileSystemEventHandler):

            def on_created(self, event):
                if not event.is_directory:
                    watcher._schedule(
                        event.src_path,
                        "created",
                    )

            def on_modified(self, event):
                if not event.is_directory:
                    watcher._schedule(
                        event.src_path,
                        "modified",
                    )

            def on_deleted(self, event):
                if not event.is_directory:
                    watcher._handle_deleted(
                        event.src_path,
                    )

            def on_moved(self, event):
                if not event.is_directory:
                    watcher._handle_deleted(
                        event.src_path,
                    )

                    watcher._schedule(
                        event.dest_path,
                        "moved",
                    )

        watch_handle = self._observer.schedule(
            _Handler(),
            key,
            recursive=True,
        )

        self._watches[key] = watch_handle

        logger.info(
            "Sekarang memantau folder: %s",
            key,
        )

    def scan_initial(self, folder: str | Path) -> None:
        """Index isi awal folder tanpa mengubah status watch."""

        root = Path(folder).resolve()

        logger.info(
            "Memulai pemindaian awal: %s",
            root,
        )

        try:
            summary = self._indexer.index_folder(root)

            logger.info(
                "Pemindaian selesai: %s | total=%d indexed=%d "
                "skipped=%d failed=%d",
                root,
                summary.get("total", 0),
                summary.get("indexed", 0),
                summary.get("skipped", 0),
                summary.get("failed", 0),
            )

        except Exception as exc:
            logger.exception(
                "Gagal melakukan pemindaian awal %s: %s",
                root,
                exc,
            )

    def unwatch(self, folder: str) -> None:
        """
        Berhenti memantau folder.

        Data yang sudah ter-index tidak dihapus.
        """
        key = str(Path(folder).resolve())

        handle = self._watches.pop(
            key,
            None,
        )

        if handle is not None and self._observer is not None:
            try:
                self._observer.unschedule(handle)
            except Exception:
                pass

            logger.info(
                "Berhenti memantau: %s",
                key,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _supported(self, path: str) -> bool:
        return (
                Path(path).suffix.lower()
                in self._exts
        )

    def _schedule(
            self,
            path: str,
            event_type: str,
    ) -> None:
        """
        Menjadwalkan index ulang dengan debounce.
        """
        if not self._supported(path):
            return

        logger.info(
            "Perubahan terdeteksi: [%s] %s",
            event_type,
            path,
        )

        with self._lock:

            existing = self._timers.pop(
                path,
                None,
            )

            if existing:
                existing.cancel()

            timer = threading.Timer(
                self._DEBOUNCE_SECONDS,
                self._do_reindex,
                args=[path],
            )

            self._timers[path] = timer

            timer.start()

    def _handle_deleted(
            self,
            path: str,
    ) -> None:
        """
        Menghapus dokumen yang sudah tidak ada dari vector store.
        """
        if not self._supported(path):
            return

        logger.info(
            "File dihapus: %s",
            path,
        )

        with self._lock:

            existing = self._timers.pop(
                path,
                None,
            )

            if existing:
                existing.cancel()

        try:
            doc_id = self._indexer._path_hash(
                Path(path)
            )

            self._indexer.store.delete_document(
                doc_id
            )

            logger.info(
                "Dihapus dari index: %s",
                path,
            )

        except Exception as exc:
            logger.warning(
                "Gagal menghapus %s dari index: %s",
                path,
                exc,
            )

    def _do_reindex(
            self,
            path: str,
    ) -> None:
        """
        Melakukan index setelah debounce selesai.
        """
        with self._lock:
            self._timers.pop(
                path,
                None,
            )

        try:
            result = self._indexer.index_file(
                path
            )

            if result:
                logger.info(
                    "Berhasil index ulang: %s",
                    path,
                )

        except Exception as exc:
            logger.warning(
                "Gagal index ulang %s: %s",
                path,
                exc,
            )