"""Word と LibreOffice による文書の変換。呼び出し側が作業用のコピーを渡し、元のファイルには触れない。"""
import os
import shutil
import subprocess
from pathlib import Path

TIMEOUT = 180
# Word の保存形式（WdSaveFormat / WdExportFormat）
_WORD_DOCX = 16
_WORD_PDF = 17


def _word(path: Path, out: Path, action: str) -> bool:
    # 開いている文書に触れないよう、別のWordで開く。文書が残っていたらWordを閉じない
    script = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0; "
        "try { $d = $w.Documents.Open($env:SIMCHECK_IN, $false, $true, $false); "
        f"{action}; $d.Close(0) }} "
        "finally { if ($w.Documents.Count -eq 0) { $w.Quit() } }"
    )
    env = {**os.environ, "SIMCHECK_IN": str(path), "SIMCHECK_OUT": str(out)}
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            env=env, capture_output=True, timeout=TIMEOUT, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.exists()


def word_to_pdf(path: Path, out: Path) -> bool:
    return _word(path, out, f"$d.ExportAsFixedFormat($env:SIMCHECK_OUT, {_WORD_PDF})")


def word_to_docx(path: Path, out: Path) -> bool:
    return _word(path, out, f"$d.SaveAs2($env:SIMCHECK_OUT, {_WORD_DOCX})")


def soffice_convert(path: Path, fmt: str) -> Path | None:
    """LibreOffice で同じフォルダに変換する。プロファイルも同じフォルダに作るので、作業フォルダで呼ぶ。"""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    try:
        subprocess.run(
            [soffice, f"-env:UserInstallation={(path.parent / 'profile').as_uri()}", "--headless", "--norestore",
             "--convert-to", fmt, "--outdir", str(path.parent), str(path)],
            capture_output=True, timeout=TIMEOUT, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = path.with_suffix(f".{fmt}")
    return out if out.exists() else None
