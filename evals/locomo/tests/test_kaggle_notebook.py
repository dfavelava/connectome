import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "kaggle.ipynb"


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return ["".join(cell["source"]) for cell in load()["cells"] if cell["cell_type"] == "code"]


def test_notebook_is_nbformat_4_without_outputs():
    notebook = load()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_code_cells_are_plain_python():
    # No ! or % magics, so every cell compiles as Python.
    for index, source in enumerate(code_cells()):
        compile(source, f"kaggle.ipynb cell {index}", "exec")


def test_default_config_is_valid():
    config: dict = {}
    exec(code_cells()[0], config)
    assert config["STAGE"] in ("extract", "answer")
    assert all(config[name].count(":") == 1 for name in ("EXTRACTOR_MODEL", "ANSWER_MODEL", "JUDGE_MODEL"))
    assert config["REPO_REF"] == "main"
    assert 0 < config["TIME_LIMIT_HOURS"] < 12
    assert config["CONCURRENCY"] >= 1
