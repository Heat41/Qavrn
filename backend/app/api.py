from __future__ import annotations

import os
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .indexer import Indexer
from .llm import OllamaClient
from .rag import RAGEngine
from .watcher import FileWatcher


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

FRONTEND_DIST = (
        Path(__file__).parent.parent.parent / "frontend" / "dist"
)


# ===========================================================================
# Lifespan
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inisialisasi komponen utama Qvarn saat aplikasi dimulai.

    Komponen:
    - Indexer
    - Ollama
    - FileWatcher

    Folder yang ada di settings.watched_folders akan:
    1. Dipindai terlebih dahulu.
    2. Kemudian dipantau secara recursive oleh Watchdog.
    """

    indexer = Indexer(settings)

    ollama = OllamaClient(
        base_url=settings.ollama_url
    )

    watcher = FileWatcher(
        indexer=indexer,
        supported_extensions=settings.supported_extensions,
    )

    watcher.start()

    # ------------------------------------------------------------------
    # Daftarkan folder yang dikonfigurasi
    # ------------------------------------------------------------------

    for folder in settings.watched_folders:

        p = Path(folder)

        if not p.exists():
            logger.warning(
                "Folder yang dikonfigurasi tidak ditemukan: %s",
                folder,
            )
            continue

        if not p.is_dir():
            logger.warning(
                "Path yang dikonfigurasi bukan folder: %s",
                folder,
            )
            continue

        try:
            watcher.watch(str(p))

        except Exception:
            logger.exception(
                "Gagal memantau folder: %s",
                p,
            )

    # ------------------------------------------------------------------
    # Simpan singleton di app.state
    # ------------------------------------------------------------------

    app.state.indexer = indexer
    app.state.ollama = ollama
    app.state.watcher = watcher

    logger.info("Qavrn API ready.")

    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    watcher.stop()


# ===========================================================================
# App
# ===========================================================================

app = FastAPI(
    title="Qavrn API",
    version="0.1.0",
    lifespan=lifespan,
)


# ===========================================================================
# CORS
# ===========================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Request Models
# ===========================================================================

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class IndexRequest(BaseModel):
    folder_path: str


class WatchRequest(BaseModel):
    folder_path: str


class DocumentRequest(BaseModel):
    file_path: str

class OpenFileRequest(BaseModel):
    file_path: str

# ===========================================================================
# Folder Explorer Helper
# ===========================================================================

def _build_folder_tree(
        root: Path,
        indexed_documents: Dict[str, Dict[str, Any]],
        watched_paths: set[str],
) -> Dict[str, Any]:
    """
    Membentuk struktur folder secara recursive.

    Struktur:

    folder
    ├── folders[]
    │   └── folder
    └── files[]
        └── file

    Folder:
    - type
    - name
    - path
    - watching
    - folders
    - files

    File:
    - type
    - name
    - path
    - extension
    - indexed
    - document
    """

    root = root.resolve()

    supported_extensions = {
        ext.lower()
        for ext in settings.supported_extensions
    }

    def build_folder(folder: Path) -> Dict[str, Any]:

        folders: List[Dict[str, Any]] = []
        files: List[Dict[str, Any]] = []

        try:
            entries = sorted(
                folder.iterdir(),
                key=lambda p: (
                    not p.is_dir(),
                    p.name.lower(),
                ),
            )

        except (PermissionError, OSError) as exc:

            logger.warning(
                "Tidak dapat membaca folder %s: %s",
                folder,
                exc,
            )

            return {
                "type": "folder",
                "name": folder.name or str(folder),
                "path": str(folder),
                "watching": str(folder) in watched_paths,
                "folders": [],
                "files": [],
                "error": "Folder tidak dapat dibaca.",
            }

        # --------------------------------------------------------------
        # Baca isi folder
        # --------------------------------------------------------------

        for entry in entries:

            try:

                # ------------------------------------------------------
                # Folder
                # ------------------------------------------------------

                if entry.is_dir():

                    folders.append(
                        build_folder(entry)
                    )

                    continue

                # ------------------------------------------------------
                # Bukan file biasa
                # ------------------------------------------------------

                if not entry.is_file():
                    continue

                # ------------------------------------------------------
                # Extension
                # ------------------------------------------------------

                extension = entry.suffix.lower()

                # Hanya tampilkan file yang didukung Qvarn.
                if extension not in supported_extensions:
                    continue

                # ------------------------------------------------------
                # Path
                # ------------------------------------------------------

                path_str = str(
                    entry.resolve()
                )

                # ------------------------------------------------------
                # Status index
                # ------------------------------------------------------

                document = indexed_documents.get(
                    path_str
                )

                # ------------------------------------------------------
                # Tambahkan file
                # ------------------------------------------------------

                files.append(
                    {
                        "type": "file",
                        "name": entry.name,
                        "path": path_str,
                        "extension": extension.lstrip("."),
                        "indexed": document is not None,
                        "document": document,
                    }
                )

            except (
                    PermissionError,
                    OSError,
            ) as exc:

                logger.warning(
                    "Tidak dapat membaca entry %s: %s",
                    entry,
                    exc,
                )

        # --------------------------------------------------------------
        # Folder result
        # --------------------------------------------------------------

        return {
            "type": "folder",
            "name": folder.name or str(folder),
            "path": str(folder),
            "watching": str(folder) in watched_paths,
            "folders": folders,
            "files": files,
        }

    return build_folder(root)


# ===========================================================================
# API — Health
# ===========================================================================

@app.get("/api/health")
async def health(
        request: Request,
) -> Dict[str, Any]:

    indexer: Indexer = request.app.state.indexer
    ollama: OllamaClient = request.app.state.ollama

    stats, ollama_ok = await asyncio.gather(
        asyncio.to_thread(
            indexer.get_stats
        ),
        asyncio.to_thread(
            ollama.is_available
        ),
    )

    return {
        "status": "ok",
        "documents": stats.documents,
        "chunks": stats.chunks,
        "ollama_available": ollama_ok,
        "model": settings.ollama_model,
    }


# ===========================================================================
# API — Ask / RAG
# ===========================================================================

@app.post("/api/ask")
async def ask(
        body: AskRequest,
        request: Request,
) -> StreamingResponse:

    indexer: Indexer = request.app.state.indexer
    ollama: OllamaClient = request.app.state.ollama

    if not body.question.strip():
        raise HTTPException(
            status_code=400,
            detail="question must not be empty",
        )

    async def event_stream():

        try:

            rag = RAGEngine(
                indexer=indexer,
                ollama=ollama,
            )

            token_iter, sources = await asyncio.to_thread(
                rag.query_stream,
                body.question,
                body.top_k,
            )

            def _next(it):
                return next(it, None)

            # ----------------------------------------------------------
            # Streaming token
            # ----------------------------------------------------------

            while True:

                token = await asyncio.to_thread(
                    _next,
                    token_iter,
                )

                if token is None:
                    break

                yield (
                        "data: "
                        + json.dumps(
                    {
                        "type": "token",
                        "content": token,
                    }
                )
                        + "\n\n"
                )

            # ----------------------------------------------------------
            # Sources
            # ----------------------------------------------------------

            sources_data = [
                {
                    "filename": s.filename,
                    "file_path": s.file_path,
                    "chunk_text": s.chunk_text,
                    "score": round(
                        s.score,
                        4,
                    ),
                }
                for s in sources
            ]

            yield (
                    "data: "
                    + json.dumps(
                {
                    "type": "sources",
                    "sources": sources_data,
                }
            )
                    + "\n\n"
            )

        except ConnectionError as exc:

            yield (
                    "data: "
                    + json.dumps(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )
                    + "\n\n"
            )

        except Exception as exc:

            logger.exception(
                "Error during /api/ask streaming"
            )

            yield (
                    "data: "
                    + json.dumps(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )
                    + "\n\n"
            )

        finally:

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ===========================================================================
# API — Index Folder
# ===========================================================================

@app.post("/api/index")
async def index_folder(
        body: IndexRequest,
        request: Request,
) -> Dict[str, Any]:

    indexer: Indexer = request.app.state.indexer

    folder = Path(
        body.folder_path
    ).resolve()

    if not folder.exists():

        raise HTTPException(
            status_code=404,
            detail=f"Folder not found: {folder}",
        )

    if not folder.is_dir():

        raise HTTPException(
            status_code=400,
            detail=f"Path bukan folder: {folder}",
        )

    summary = await asyncio.to_thread(
        indexer.index_folder,
        folder,
    )

    stats = await asyncio.to_thread(
        indexer.get_stats
    )

    return {
        **summary,
        "documents": stats.documents,
        "chunks": stats.chunks,
    }


# ===========================================================================
# API — Stats
# ===========================================================================

@app.get("/api/stats")
async def stats(
        request: Request,
) -> Dict[str, Any]:

    indexer: Indexer = request.app.state.indexer

    s = await asyncio.to_thread(
        indexer.get_stats
    )

    return {
        "documents": s.documents,
        "chunks": s.chunks,
        "storage_mb": round(
            s.storage_mb,
            2,
        ),
        "storage_bytes": s.storage_bytes,
    }


# ===========================================================================
# API — Indexed Documents
# ===========================================================================

@app.get("/api/documents")
async def list_documents(
        request: Request,
) -> Dict[str, Any]:

    indexer: Indexer = request.app.state.indexer

    docs = await asyncio.to_thread(
        indexer.store.list_documents
    )

    return {
        "documents": docs
    }


# ===========================================================================
# API — Folder Explorer
# ===========================================================================

@app.get("/api/folders")
async def get_folder_tree(
        request: Request,
) -> Dict[str, Any]:
    """
    Mengembalikan struktur folder dan file
    secara recursive.

    Hanya root folder yang sedang dipantau
    yang ditampilkan.

    Subfolder dan file di bawah root akan
    ditampilkan secara recursive.
    """

    indexer: Indexer = request.app.state.indexer
    watcher: FileWatcher = request.app.state.watcher

    # --------------------------------------------------------------
    # Folder yang sedang dipantau
    # --------------------------------------------------------------

    watched = watcher.watched_folders

    watched_paths = {
        str(
            Path(folder).resolve()
        )
        for folder in watched
    }

    # --------------------------------------------------------------
    # Ambil dokumen yang sudah ter-index
    # --------------------------------------------------------------

    documents = await asyncio.to_thread(
        indexer.store.list_documents
    )

    indexed_documents: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for document in documents:

        file_path = document.get(
            "file_path"
        )

        if not file_path:
            continue

        try:

            key = str(
                Path(file_path).resolve()
            )

            indexed_documents[key] = document

        except (
                OSError,
                ValueError,
        ):
            continue

    # --------------------------------------------------------------
    # Build semua root folder
    # --------------------------------------------------------------

    roots: List[Dict[str, Any]] = []

    for watched_folder in watched:

        root = Path(
            watched_folder
        ).resolve()

        if not root.exists():
            continue

        if not root.is_dir():
            continue

        roots.append(
            _build_folder_tree(
                root=root,
                indexed_documents=indexed_documents,
                watched_paths=watched_paths,
            )
        )

    return {
        "folders": roots
    }


# ===========================================================================
# API — Scan Single Document
# ===========================================================================

@app.post("/api/documents/scan")
async def scan_document(
        body: DocumentRequest,
        request: Request,
) -> Dict[str, Any]:

    indexer: Indexer = request.app.state.indexer

    file_path = Path(
        body.file_path
    ).resolve()

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail=f"File tidak ditemukan: {file_path}",
        )

    if not file_path.is_file():

        raise HTTPException(
            status_code=400,
            detail=f"Path bukan file: {file_path}",
        )

    supported_extensions = {
        ext.lower()
        for ext in settings.supported_extensions
    }

    if file_path.suffix.lower() not in supported_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Format file tidak didukung: "
                f"{file_path.suffix}"
            ),
        )

    indexed = await asyncio.to_thread(
        indexer.index_file,
        file_path,
    )

    stats = await asyncio.to_thread(
        indexer.get_stats
    )

    return {
        "success": True,
        "file_path": str(file_path),
        "filename": file_path.name,
        "indexed": indexed,
        "message": (
            "File berhasil dipindai dan di-index."
            if indexed
            else "File sudah terbaru. Tidak perlu di-index ulang."
        ),
        "documents": stats.documents,
        "chunks": stats.chunks,
    }

# ===========================================================================
# API — Open File
# ===========================================================================

@app.post("/api/open-file")
def open_file(request: OpenFileRequest):
    """Membuka file lokal menggunakan aplikasi default Windows."""

    file_path = os.path.abspath(request.file_path)

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File tidak ditemukan: {file_path}",
        )

    try:
        os.startfile(file_path)
        return {
            "status": "ok",
            "message": "File berhasil dibuka",
            "file_path": file_path,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membuka file: {exc}",
        )

# ===========================================================================
# API — Reindex Single Document
# ===========================================================================

@app.post("/api/documents/reindex")
async def reindex_document(
        body: DocumentRequest,
        request: Request,
) -> Dict[str, Any]:

    indexer: Indexer = request.app.state.indexer

    file_path = Path(
        body.file_path
    ).resolve()

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail=f"File tidak ditemukan: {file_path}",
        )

    if not file_path.is_file():

        raise HTTPException(
            status_code=400,
            detail=f"Path bukan file: {file_path}",
        )

    supported_extensions = {
        ext.lower()
        for ext in settings.supported_extensions
    }

    if file_path.suffix.lower() not in supported_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Format file tidak didukung: "
                f"{file_path.suffix}"
            ),
        )

    # --------------------------------------------------------------
    # Hapus index lama
    # --------------------------------------------------------------

    doc_id = indexer._path_hash(
        file_path
    )

    await asyncio.to_thread(
        indexer.store.delete_document,
        doc_id,
    )

    # --------------------------------------------------------------
    # Index ulang
    # --------------------------------------------------------------

    indexed = await asyncio.to_thread(
        indexer.index_file,
        file_path,
    )

    stats = await asyncio.to_thread(
        indexer.get_stats
    )

    return {
        "success": indexed,
        "file_path": str(file_path),
        "filename": file_path.name,
        "indexed": indexed,
        "message": (
            "File berhasil di-index ulang."
            if indexed
            else "File gagal di-index ulang."
        ),
        "documents": stats.documents,
        "chunks": stats.chunks,
    }


# ===========================================================================
# API — Watch Folder
# ===========================================================================

@app.post("/api/watch")
async def watch_folder(
        body: WatchRequest,
        request: Request,
) -> Dict[str, Any]:

    indexer: Indexer = request.app.state.indexer
    watcher: FileWatcher = request.app.state.watcher

    folder = Path(
        body.folder_path
    ).resolve()

    if not folder.exists():

        raise HTTPException(
            status_code=404,
            detail=f"Folder not found: {folder}",
        )

    if not folder.is_dir():

        raise HTTPException(
            status_code=400,
            detail=f"Path bukan folder: {folder}",
        )

    # --------------------------------------------------------------
    # Index folder terlebih dahulu
    # --------------------------------------------------------------

    summary = await asyncio.to_thread(
        indexer.index_folder,
        folder,
    )

    stats = await asyncio.to_thread(
        indexer.get_stats
    )

    # --------------------------------------------------------------
    # Aktifkan watcher
    # --------------------------------------------------------------

    await asyncio.to_thread(
        watcher.watch,
        str(folder),
    )

    return {
        **summary,
        "documents": stats.documents,
        "chunks": stats.chunks,
        "watching": True,
        "folder": str(folder),
    }


# ===========================================================================
# API — Watched Folders
# ===========================================================================

@app.get("/api/watched")
async def get_watched(
        request: Request,
) -> Dict[str, List[str]]:

    watcher: FileWatcher = request.app.state.watcher

    return {
        "folders": watcher.watched_folders
    }


# ===========================================================================
# API — Unwatch Folder
# ===========================================================================

@app.delete("/api/watch")
async def unwatch_folder(
        body: WatchRequest,
        request: Request,
) -> Dict[str, Any]:

    watcher: FileWatcher = request.app.state.watcher

    await asyncio.to_thread(
        watcher.unwatch,
        body.folder_path,
    )

    return {
        "folder": body.folder_path,
        "watching": False,
    }


# ===========================================================================
# Serve Frontend
# ===========================================================================

if FRONTEND_DIST.exists():

    assets_dir = FRONTEND_DIST / "assets"

    if assets_dir.exists():

        app.mount(
            "/assets",
            StaticFiles(
                directory=str(assets_dir)
            ),
            name="assets",
        )

    @app.get(
        "/{full_path:path}",
        include_in_schema=False,
    )
    async def serve_spa(
            full_path: str,
    ):

        candidate = (
                FRONTEND_DIST / full_path
        )

        if (
                candidate.exists()
                and candidate.is_file()
        ):
            return FileResponse(
                str(candidate)
            )

        return FileResponse(
            str(
                FRONTEND_DIST
                / "index.html"
            )
        )

else:

    @app.get(
        "/",
        include_in_schema=False,
    )
    async def no_frontend():

        return {
            "message": (
                "Frontend not built. "
                "Run: cd frontend && "
                "npm install && npm run build"
            ),
            "api_docs": "/docs",
        }