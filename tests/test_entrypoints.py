from pathlib import Path
import subprocess
import sys


def test_training_script_supports_direct_file_execution():
    root = Path(__file__).resolve().parents[1]
    script = root / "nn" / "train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-iter" in result.stdout
