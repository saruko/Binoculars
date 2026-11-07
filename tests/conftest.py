import pytest

from simcheck import apikeys


@pytest.fixture(autouse=True)
def no_saved_api_keys(monkeypatch):
    # 利用者が資格情報マネージャーに保存したキーをテストで使わない
    monkeypatch.setattr(apikeys, "_keyring", lambda: None)
