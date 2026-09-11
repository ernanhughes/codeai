from codeai.cli import main


def test_run_create_list_show(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["run", "create", "Fix the flaky writer-runtime test"]) == 0
    created = capsys.readouterr().out
    run_id = next(line.split(": ", 1)[1] for line in created.splitlines() if line.startswith("run_id: "))

    assert main(["run", "list"]) == 0
    listed = capsys.readouterr().out
    assert run_id in listed

    assert main(["run", "show", run_id]) == 0
    shown = capsys.readouterr().out
    assert "Fix the flaky writer-runtime test" in shown
