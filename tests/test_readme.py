"""
Runs the README Quick Start exactly as written, against the fake OpenAI server in examples/.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from fake_openai import serve  # noqa: E402

REPORT = """# Annual report

Marie Curie moved to Paris in 1891 to study physics. She shared the Nobel Prize in 1903.

In 1911 she received a second Nobel Prize, in chemistry, for discovering polonium and radium.
"""


def quick_start_code() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Quick Start", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def run_quick_start(tmp_path: Path, base_url: str) -> subprocess.CompletedProcess:
    (tmp_path / "quickstart.py").write_text(quick_start_code(), encoding="utf-8")
    (tmp_path / "report.md").write_text(REPORT, encoding="utf-8")
    env = dict(os.environ, OPENAI_BASE_URL=base_url, OPENAI_API_KEY="unused")
    return subprocess.run(
        [sys.executable, "quickstart.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def fake_url():
    server, url = serve()
    yield url
    server.shutdown()


def test_quick_start_runs_as_written(tmp_path, fake_url):
    result = run_quick_start(tmp_path, fake_url)

    assert result.returncode == 0, result.stderr
    assert "--- Chunk" in result.stdout
    assert "Entities: " in result.stdout
    assert "Entities: none" not in result.stdout
    assert "Summary: " in result.stdout


def test_quick_start_reports_unreachable_endpoint(tmp_path):
    result = run_quick_start(tmp_path, "http://127.0.0.1:9/v1")

    assert result.returncode != 0
    assert "LLMBackendError: Could not reach the LLM endpoint at http://127.0.0.1:9/v1" in (
        result.stderr
    )
