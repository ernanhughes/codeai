from __future__ import annotations

import subprocess
import sys
from collections import Counter

from codeai.corpus_v2 import CORPUS2_VERSION, get_v2_task, semantic_corpus


def test_v2_corpus_has_four_strata_of_ten():
    corpus = semantic_corpus()
    assert CORPUS2_VERSION == "semantic-repair-v1"
    assert len(corpus) == 40
    assert Counter(t.stratum for t in corpus) == {
        "local": 10,
        "semantic": 10,
        "architectural": 10,
        "boundary": 10,
    }
    assert len({t.task_id for t in corpus}) == 40
    assert all(t.family for t in corpus)


def test_v2_statements_hide_fault_labels():
    corpus = semantic_corpus()
    diagnosis_words = ("bug ", "bug.", "fault", "mistake", "broken abstraction",
                       "wrong abstraction", "stale-state bug", "should be deleted")
    for task in corpus:
        prompt = task.problem_statement.lower()
        assert not any(word in prompt for word in diagnosis_words), task.task_id
        assert task.hidden_tests not in task.problem_statement
        assert task.reference_solution not in task.problem_statement


def test_v2_hidden_tests_fail_on_starter_and_pass_on_reference(tmp_path):
    for task in semantic_corpus():
        for label, code, expect_ok in (
            ("starter", task.starter_code, False),
            ("reference", task.reference_solution, True),
        ):
            workdir = tmp_path / task.task_id / label
            workdir.mkdir(parents=True)
            for name, content in task.support_files:
                (workdir / name).write_text(content)
            (workdir / "candidate.py").write_text(code)
            (workdir / "hidden_test.py").write_text(task.hidden_tests)
            completed = subprocess.run(
                [sys.executable, "hidden_test.py"],
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
            )
            assert (completed.returncode == 0) == expect_ok, (task.task_id, label)


def test_v2_lookup():
    assert get_v2_task("v2-lsp-square").stratum == "architectural"
