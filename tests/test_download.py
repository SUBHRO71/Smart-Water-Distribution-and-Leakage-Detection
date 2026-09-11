import hashlib
import io
import json

import pytest

from smart_water import data


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_checks_integrity_and_skips_verified_files(tmp_path, monkeypatch, corrupt):
    payload = b"publisher data"
    names = ["README.txt", "L-TOWN.inp", "2018_SCADA_Flows.csv"]
    checksum = "md5:" + hashlib.md5(payload).hexdigest()
    metadata = {"metadata": {"license": {"id": "cc-by-4.0"}},
                "files": [{"key": name, "checksum": checksum} for name in names]}
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        if url == data.RECORD:
            return io.BytesIO(json.dumps(metadata).encode())
        return io.BytesIO(b"broken" if corrupt else payload)

    monkeypatch.setattr(data, "urlopen", open_url)
    if corrupt:
        with pytest.raises(ValueError, match="Checksum mismatch"):
            data.download(tmp_path)
        assert not list(tmp_path.glob("*.part"))
        assert not (tmp_path / "README.txt").exists()
    else:
        assert len(data.download(tmp_path)) == 3
        calls.clear()
        assert len(data.download(tmp_path)) == 3
        assert calls == [data.RECORD]
        assert (tmp_path / "manifest_2018.json").exists()
