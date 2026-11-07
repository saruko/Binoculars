"""APIキーの保存と、DOC・RTF・ODT と文字コードの読み込み。"""
import docx
import pytest

from simcheck import apikeys, library, office, pdfreport, revise
from simcheck.extract import decode_text, extract_document

JA_TEXT = "加齢黄斑変性は高齢者における視力低下の主要な原因であり、抗血管内皮増殖因子療法によって治療成績は大きく改善した。" * 3


class FakeKeyring:
    def __init__(self):
        self.saved = {}

    def get_password(self, service, name):
        return self.saved.get((service, name))

    def set_password(self, service, name, value):
        self.saved[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.saved:
            raise KeyError(name)
        del self.saved[(service, name)]


@pytest.fixture
def keyring(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(apikeys, "_keyring", lambda: fake)
    for name in apikeys.KEYS:
        monkeypatch.delenv(name, raising=False)
    return fake


def test_saved_key_is_used_and_environment_wins(keyring, monkeypatch):
    assert apikeys.get("BRAVE_API_KEY") is None
    apikeys.save("BRAVE_API_KEY", "  saved  ")
    assert apikeys.get("BRAVE_API_KEY") == "saved"
    assert apikeys.status()["BRAVE_API_KEY"] == "保存済み"
    monkeypatch.setenv("BRAVE_API_KEY", "from-env")
    assert apikeys.get("BRAVE_API_KEY") == "from-env"
    assert apikeys.status()["BRAVE_API_KEY"] == "環境変数"
    apikeys.delete("BRAVE_API_KEY")
    apikeys.delete("BRAVE_API_KEY")  # 保存されていなくても失敗しない
    monkeypatch.delenv("BRAVE_API_KEY")
    assert apikeys.status()["BRAVE_API_KEY"] == "未設定"


def test_save_rejects_unknown_names_and_empty_values(keyring):
    with pytest.raises(ValueError):
        apikeys.save("OTHER_KEY", "x")
    with pytest.raises(ValueError):
        apikeys.save("GEMINI_API_KEY", "  ")


def test_without_keyring_only_environment_is_used(monkeypatch):
    monkeypatch.setattr(apikeys, "_keyring", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert apikeys.get("GEMINI_API_KEY") is None
    with pytest.raises(apikeys.KeyStoreUnavailable):
        apikeys.save("GEMINI_API_KEY", "x")


def test_broken_keyring_backend_is_ignored(monkeypatch):
    class Broken:
        def get_password(self, *args):
            raise RuntimeError("no backend")

    monkeypatch.setattr(apikeys, "_keyring", lambda: Broken())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert apikeys.get("GEMINI_API_KEY") is None


def test_local_server_uses_saved_key(keyring):
    apikeys.save("STRATA_API_KEY", "secret")
    assert revise._headers(revise.PROVIDERS["strata"]) == {"Authorization": "Bearer secret"}


@pytest.mark.parametrize("encoding", ["shift_jis", "euc_jp", "utf-8", "utf-16"])
def test_text_encodings_are_detected(encoding):
    assert decode_text(JA_TEXT.encode(encoding)) == JA_TEXT


def test_english_cp1252_text_is_readable():
    text = "The patient’s naïve café visit — résumé attached. " * 5
    assert decode_text(text.encode("cp1252")) == text


def _fake_word(created, fail=False):
    def convert(path, out):
        created.append(path.name)
        if fail:
            return False
        d = docx.Document()
        d.add_paragraph("converted body text")
        d.save(out)
        return True
    return convert


@pytest.mark.parametrize("suffix", [".doc", ".rtf", ".odt"])
def test_word_formats_are_converted_from_a_copy(tmp_path, monkeypatch, suffix):
    created = []
    monkeypatch.setattr(office, "word_to_docx", _fake_word(created))
    path = tmp_path / f"原稿{suffix}"
    path.write_bytes(b"original bytes")
    extracted = extract_document(path)
    assert extracted.text == "converted body text"
    assert extracted.title == "原稿"
    assert created == [f"input{suffix}"]
    assert path.read_bytes() == b"original bytes"


def test_conversion_without_word_or_libreoffice_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(office, "word_to_docx", _fake_word([], fail=True))
    monkeypatch.setattr(office, "soffice_convert", lambda path, fmt: None)
    path = tmp_path / "old.doc"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="Word か LibreOffice"):
        extract_document(path)


def test_library_skips_formats_that_need_word(tmp_path, monkeypatch):
    (tmp_path / "a.doc").write_bytes(b"x")
    (tmp_path / "b.rtf").write_bytes(b"x")
    seen = []

    class Store:
        def has_ext_id(self, ext_id):
            return False

    monkeypatch.setattr(library, "sha256_file", lambda p: seen.append(p.name) or p.name)
    library.add_library(Store(), tmp_path, progress=lambda message: None)
    assert seen == []


def test_pdf_report_converts_rtf_with_word(tmp_path, monkeypatch):
    inputs = []
    monkeypatch.setattr(pdfreport, "_word_to_pdf", lambda path, out: inputs.append(path.name) or False)
    monkeypatch.setattr(pdfreport, "_soffice_to_pdf", lambda path: None)
    path = tmp_path / "m.rtf"
    path.write_bytes(b"x")
    _, notices = pdfreport._open_base(path, "body")
    assert inputs == ["input.rtf"]
    assert any("組み直したもの" in n for n in notices)
