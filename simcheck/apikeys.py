"""APIキーの読み出しと保存。環境変数を優先し、なければ Windows 資格情報マネージャーから読む。"""
import os

SERVICE = "binoculars-simcheck"
KEYS = {
    "GEMINI_API_KEY": "Gemini API",
    "ANTHROPIC_API_KEY": "Claude API",
    "BRAVE_API_KEY": "Brave Search API（Web照合）",
    "STRATA_API_KEY": "Strata",
    "LLAMA_API_KEY": "llama.cpp",
}


class KeyStoreUnavailable(RuntimeError):
    pass


def _keyring():
    # GitHub Actions では入れていないので、なければ環境変数だけを使う
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def _check_name(name: str):
    if name not in KEYS:
        raise ValueError(f"未対応のAPIキーです: {name}")


def stored(name: str) -> str | None:
    store = _keyring()
    if store is None:
        return None
    try:
        return (store.get_password(SERVICE, name) or "").strip() or None
    except Exception:
        # 資格情報の保存先がない環境（Linux のサーバーなど）では使わない
        return None


def get(name: str) -> str | None:
    return os.environ.get(name, "").strip() or stored(name)


def save(name: str, value: str):
    _check_name(name)
    value = value.strip()
    if not value:
        raise ValueError("APIキーが空です。")
    store = _keyring()
    if store is None:
        raise KeyStoreUnavailable("keyring が入っていないため保存できません。環境変数で設定してください。")
    try:
        store.set_password(SERVICE, name, value)
    except Exception as e:
        raise KeyStoreUnavailable(f"資格情報マネージャーに保存できませんでした: {e}") from e


def delete(name: str):
    _check_name(name)
    store = _keyring()
    if store is None:
        return
    try:
        store.delete_password(SERVICE, name)
    except Exception:
        # 保存されていないときも例外になる
        pass


def status() -> dict[str, str]:
    """各キーの設定元（環境変数・保存済み・未設定）。値そのものは返さない。"""
    result = {}
    for name in KEYS:
        if os.environ.get(name, "").strip():
            result[name] = "環境変数"
        elif stored(name):
            result[name] = "保存済み"
        else:
            result[name] = "未設定"
    return result
