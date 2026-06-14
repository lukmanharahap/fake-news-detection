from __future__ import annotations
import re


class TextCleaner:
    """Helper for text normalization and source-specific cleanup."""

    @staticmethod
    def extract_narasi_only(text: str) -> str:
        if not isinstance(text, str):
            return ""
        pattern = r"\[?narasi\]?\W*(.*?)(?=\n\s*={3,}|\n\s*\[|\Z|penjelasan)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'^[“"\'”]|[“"\'”]$', '', extracted)
            return extracted.strip()

        return ""

    @staticmethod
    def clean_text_basic(text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'@\w+', '', cleaned)
        cleaned = re.sub(r'#\w+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip().lower()

    @staticmethod
    def clean_cnn(text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = text.lower()
        cleaned = re.sub(r"\(?[a-z]{3,7}/[a-z]{3}/[a-z]{3}\)?", "", cleaned)
        cleaned = re.sub(r"\(?[a-z]{3,7}/[a-z]{3}\)?", "", cleaned)
        cleaned = re.sub(r"baca\s*(berita)?\s*(se)?lengkapnya\s*di\s*sini", "", cleaned)
        cleaned = re.sub(r"baca\s*juga\W*", "", cleaned)
        cleaned = re.sub(r"[a-z\s]*cnnindonesia", "", cleaned)
        cleaned = re.sub(r"com,.*?\(\d*/\d*\)", "", cleaned)
        cleaned = re.sub(r"\(cnn\s*[a-z\s]+\)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r" ,", ",", cleaned)
        cleaned = re.sub(r" \.", ".", cleaned)
        return cleaned.strip()

    @staticmethod
    def clean_kompas(text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = text.lower()
        cleaned = re.sub(
            r"(dikutip|berdasarkan)\s*(dari|pantauan)\s*kompas", "", cleaned
        )
        cleaned = re.sub(r"baca\s*juga\W*", "", cleaned)
        cleaned = re.sub(r"com,\s*selasa", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r" ,", ",", cleaned)
        cleaned = re.sub(r" \.", ".", cleaned)
        return cleaned.strip()

    @staticmethod
    def clean_tempo(text: str) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = text.lower()

        journalist_names = [
            "m julnis firmansyah", "julnis firmansyah", "sapri maulana",
            "hendrik khoirul muhid", "septhia ryanthie", "mutia yuantisya",
            "pribadi wicaksono", "alfitria nefi pratiwi", "gadis oktaviani",
            "muh raihan muzakki", "risma damayanti", "rahma dwi safitri",
            "naufal ridhwan aly", "annisa firdausi", "fathur rachman",
            "arrijal rachman", "ima dini shafira", "fajar pebrianto",
            "faiz zaki", "delfi ana harahap", "nugroho catur pamungkas",
            "dewi nurita", "mirza bagaskara"
        ]
        journalist_names.sort(key=len, reverse=True)
        escaped_names = [re.escape(name) for name in journalist_names]
        journalist_regex = re.compile(r'\b(' + '|'.join(escaped_names) + r')\b', flags=re.IGNORECASE)

        cleaned = journalist_regex.sub("", cleaned)
        cleaned = re.sub(r"(nurita|firmansyah|yuantisya)?ikuti(.*)?di\s*sini", "", cleaned)
        cleaned = re.sub(r"co(.*)?di\s*sini", "", cleaned)
        cleaned = re.sub(r"12(3)?\s*selanjutnya", "", cleaned)
        cleaned = re.sub(r"baca\s*juga\W*", "", cleaned)
        cleaned = re.sub(r"(ikuti)?\W*berita\W*terkini\W*dari\W*tempo\W*[A-Za-z0-9\s]+\b", "", cleaned)
        cleaned = re.sub(r"\bikuti\W*klik\W*di\W*sini", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r" ,", ",", cleaned)
        cleaned = re.sub(r" \.", ".", cleaned)
        return cleaned.strip()

    @staticmethod
    def clean_hoax(text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = text.lower()
        cleaned = re.sub(r"selengkapnya\s*(terdapat|ada)?\s*di\W*", "", cleaned)
        cleaned = re.sub(r"tersebut\s*tidak\s*benar\W*", "", cleaned)
        cleaned = re.sub(r"selengkapnya\s*pada\W*", "", cleaned)
        cleaned = re.sub(r"\W*(lanjutan)?\W*narasi\W*", "", cleaned)
        cleaned = re.sub(r"(diterjemahkan)?\W*ke\W*(dalam)?\W*bahasa\W*", "", cleaned)
        cleaned = re.sub(r"(gambar|foto|hoax|sumber|akun|facebook|akun\s*facebook|tidak\s*benar|terjemahan|video|judul|faktanya)", "", cleaned)
        cleaned = re.sub(r"(berita|beredar|postingan|unggah|the|yang\s*salah|ternyata|bagian|palsu|artikel)", "", cleaned)
        cleaned = re.sub(r"=", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r" ,", ",", cleaned)
        cleaned = re.sub(r" \.", ".", cleaned)
        return cleaned.strip()
