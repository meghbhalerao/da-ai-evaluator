from pathlib import Path

def find_git_root(path: Path = Path(__file__).resolve()) -> Path:
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("No .git directory found in parent directories.")

