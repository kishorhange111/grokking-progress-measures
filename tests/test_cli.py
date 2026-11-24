import json

from grok.cli import main, sweep_table


def test_sweep_end_to_end(tmp_path):
    main(
        [
            "sweep",
            "--p",
            "11",
            "--d",
            "16",
            "--steps",
            "60",
            "--eval-every",
            "20",
            "--fracs",
            "0.5",
            "--wds",
            "0",
            "1",
            "--workers",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    rows = json.loads((tmp_path / "sweep.json").read_text())
    assert len(rows) == 2 and {r["config"]["wd"] for r in rows} == {0.0, 1.0}
    assert (tmp_path / "sweep.png").exists()
    assert "| 0.5 | 1.0 | 0 |" in sweep_table(rows)


def test_single_writes_plot(tmp_path):
    main(
        [
            "single",
            "--p",
            "11",
            "--d",
            "16",
            "--steps",
            "40",
            "--eval-every",
            "20",
            "--out",
            str(tmp_path),
        ]
    )
    assert list(tmp_path.glob("single_*.png")) and list(tmp_path.glob("single_*.json"))
