import json
from pathlib import Path

from locomo_eval.extraction import (
    EXTRACT_V2_VERSION,
    EXTRACT_VERSION,
    LIFECYCLE_V2_VERSION,
    LIFECYCLE_V3_VERSION,
    LIFECYCLE_V4_VERSION,
    LIFECYCLE_VERSION,
)
from locomo_eval.prompts import (
    ANSWER_V2_VERSION,
    ANSWER_VERSION,
    JUDGE_V2_VERSION,
    JUDGE_V3_VERSION,
    JUDGE_VERSION,
    load_prompt,
)

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
    assert config["EXTRACT_PROMPT"] == EXTRACT_VERSION
    assert load_prompt(config["EXTRACT_PROMPT"]).version in (EXTRACT_VERSION, EXTRACT_V2_VERSION, LIFECYCLE_VERSION, LIFECYCLE_V2_VERSION, LIFECYCLE_V3_VERSION, LIFECYCLE_V4_VERSION)
    assert config["ANSWER_PROMPT"] == ANSWER_VERSION
    assert load_prompt(config["ANSWER_PROMPT"]).version in (ANSWER_VERSION, ANSWER_V2_VERSION)
    assert config["JUDGE_PROMPT"] == JUDGE_VERSION
    assert load_prompt(config["JUDGE_PROMPT"]).version in (JUDGE_VERSION, JUDGE_V2_VERSION, JUDGE_V3_VERSION)
    assert config["ANSWER_K"] is None and config["ANSWER_THINK"] is False
    assert 0 < config["TIME_LIMIT_HOURS"] < 12
    assert config["CONCURRENCY"] >= 1
