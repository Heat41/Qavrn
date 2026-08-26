from __future__ import annotations

import json
import logging
from typing import Iterator

from .config import settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """
Anda adalah Qvarn, asisten AI untuk menjawab pertanyaan berdasarkan dokumen lokal.

ATURAN UTAMA:

1. Gunakan HANYA informasi dari CONTEXT.
2. Jangan menggunakan pengetahuan dari luar CONTEXT.
3. Jangan mengarang, menebak, atau menyimpulkan angka yang tidak tertulis jelas.
4. Jika informasi yang diminta tidak ada secara jelas dalam CONTEXT, jawab:
   "Informasi tersebut tidak ditemukan dalam dokumen yang diberikan."
5. Jawab langsung, singkat, dan dalam Bahasa Indonesia.
6. Untuk angka, gunakan persis seperti yang tertulis di dokumen.
7. Jangan mengubah angka menjadi angka lain.
8. Jangan menganggap angka 0 sebagai jawaban hanya karena angka 0 muncul di dokumen.
9. Jika terdapat beberapa dokumen atau beberapa masa pajak, jangan mencampurkan nilainya.
10. Jika pertanyaan menyebut nama file tertentu, gunakan hanya informasi dari file tersebut.
11. Jika pertanyaan menyebut tahun tetapi tidak menyebut masa pajak, perhatikan semua dokumen yang tersedia pada tahun tersebut.
12. Jika terdapat beberapa masa pajak dalam tahun yang sama dan nilai berbeda, jangan memilih satu nilai secara sembarangan.
13. Untuk tabel, cocokkan BARIS dengan KOLOM sebelum mengambil nilai.
14. Jangan menukar nilai antar kolom.
15. Jika dokumen memberikan hasil "Selisih", gunakan nilai tersebut.
16. Jika informasi halaman tersedia dalam CONTEXT, sebutkan halaman bila relevan.
17. Untuk pertanyaan lokasi dokumen, gunakan File Path yang tersedia dalam CONTEXT.
18. Jangan mengambil informasi dari Source lain di luar CONTEXT.

KHUSUS DOKUMEN PAJAK:

- Bedakan SPT dengan Tanda Terima/Bukti Penerimaan Elektronik.
- Tanda Terima digunakan untuk informasi penerimaan SPT, bukan sebagai sumber utama nilai yang terdapat di dalam SPT.
- Untuk nilai PPN, DPP, PPh, kurang bayar, lebih bayar, atau angka pajak lainnya, prioritaskan bagian SPT yang memuat tabel atau nilai tersebut.
- Jika SPT berstatus Nihil dan dokumen secara jelas menunjukkan nilai 0 untuk informasi yang ditanyakan, maka jawaban 0 adalah benar.
- Jangan mengambil angka dari dokumen lain hanya karena dokumen tersebut memiliki tahun yang sama.
- Jika dua angka muncul pada baris yang sama dengan kolom "Menurut WP/SPT" dan "Menurut Pemeriksa", cocokkan berdasarkan posisi kolom.
- Jika pertanyaan meminta "Menurut Wajib Pajak" dan "Menurut Pemeriksa", jawab kedua nilai tersebut secara eksplisit.
- Jika pertanyaan meminta selisih, hitung hanya jika kedua nilai tersedia dengan jelas dalam CONTEXT.

FORMAT:

Untuk lokasi:

Dokumen tersebut berada di:
C:\\path\\folder

Untuk perbandingan:

Menurut WP/SPT: Rp X
Menurut Pemeriksa: Rp Y

Untuk selisih:

Selisih: Rp Z
""".strip()


class OllamaClient:
    """Thin client for Ollama lokal."""

    def __init__(
            self,
            base_url: str | None = None,
    ) -> None:

        self.base_url = (
                base_url or settings.ollama_url
        ).rstrip("/")

    # ------------------------------------------------------------------
    # AVAILABLE
    # ------------------------------------------------------------------

    def is_available(self) -> bool:

        try:
            import requests

            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=3,
            )

            return resp.status_code == 200

        except Exception:
            return False

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------

    def generate(
            self,
            prompt: str,
            context: str = "",
            model: str | None = None,
    ) -> str:

        model = (
                model
                or settings.ollama_model
        )

        return "".join(
            self._stream(
                prompt,
                context,
                model,
            )
        )

    # ------------------------------------------------------------------
    # GENERATE STREAM
    # ------------------------------------------------------------------

    def generate_stream(
            self,
            prompt: str,
            context: str = "",
            model: str | None = None,
    ) -> Iterator[str]:

        model = (
                model
                or settings.ollama_model
        )

        yield from self._stream(
            prompt,
            context,
            model,
        )

    # ------------------------------------------------------------------
    # BUILD BODY
    # ------------------------------------------------------------------

    def _build_body(
            self,
            prompt: str,
            context: str,
            model: str,
    ) -> dict:

        if context:

            user_content = (
                "CONTEXT:\n"
                f"{context}\n\n"
                "PERTANYAAN:\n"
                f"{prompt}\n\n"
                "Jawab hanya berdasarkan CONTEXT. "
                "Jawab singkat dan langsung."
            )

        else:

            user_content = (
                "CONTEXT DOKUMEN LOKAL tidak tersedia.\n\n"
                f"PERTANYAAN PENGGUNA:\n{prompt}\n\n"
                "Karena tidak ada context dokumen, "
                "jangan mengarang jawaban."
            )

        return {
            "model": model,
            "system": _SYSTEM_PROMPT,
            "prompt": user_content,
            "stream": True,

            "options": {
                # CPU only
                "num_gpu": 0,

                # Context cukup untuk pertanyaan dokumen
                "num_ctx": 2048,

                # Jawaban pendek = lebih cepat
                "num_predict": 96,

                # Rendah supaya angka lebih stabil
                "temperature": 0.1,
            },
        }

    # ------------------------------------------------------------------
    # STREAM
    # ------------------------------------------------------------------

    def _stream(
            self,
            prompt: str,
            context: str,
            model: str,
    ) -> Iterator[str]:

        try:
            import requests

        except ImportError:

            raise ImportError(
                "requests is required: pip install requests"
            )

        body = self._build_body(
            prompt,
            context,
            model,
        )

        url = (
            f"{self.base_url}/api/generate"
        )

        try:

            with requests.post(
                    url,
                    json=body,
                    stream=True,

                    # connect timeout = 10 detik
                    # read timeout = 300 detik
                    #
                    # CPU-only Qwen bisa lama.
                    timeout=(10, 300),

            ) as resp:

                if resp.status_code >= 400:

                    logger.error(
                        "Ollama error response: %s",
                        resp.text,
                    )

                    resp.raise_for_status()

                for raw_line in resp.iter_lines():

                    if not raw_line:
                        continue

                    try:

                        data = json.loads(
                            raw_line
                        )

                    except json.JSONDecodeError as exc:

                        logger.warning(
                            "Undecodable line from Ollama: %s — %s",
                            raw_line,
                            exc,
                        )

                        continue

                    token = data.get(
                        "response",
                        "",
                    )

                    if token:
                        yield token

                    if data.get("done"):
                        break

        except requests.exceptions.ConnectTimeout:

            raise ConnectionError(
                f"Ollama tidak dapat dihubungi "
                f"dalam batas waktu koneksi: "
                f"{self.base_url}"
            )

        except requests.exceptions.ReadTimeout:

            raise TimeoutError(
                "Ollama terlalu lama menghasilkan jawaban. "
                "Karena Qvarn berjalan menggunakan CPU, "
                "proses generasi dapat membutuhkan waktu cukup lama."
            )

        except requests.exceptions.ConnectionError:

            raise ConnectionError(
                f"Tidak dapat terhubung ke Ollama di "
                f"{self.base_url}. "
                "Pastikan Ollama sedang berjalan."
            )

        except requests.exceptions.HTTPError as exc:

            raise RuntimeError(
                f"Ollama mengembalikan error: {exc}"
            )