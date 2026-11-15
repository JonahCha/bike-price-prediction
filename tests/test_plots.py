from bike_price_prediction import run
from plots import save_console_image, save_figures
from tests.test_bike_price_prediction import make_bike_csv, REF_YEAR


def test_figures_are_written(tmp_path):
    summary = run(make_bike_csv(tmp_path / "bikes.csv"), reference_year=REF_YEAR)
    paths = save_figures(summary, tmp_path / "images")
    assert len(paths) == 5
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_console_image_is_written(tmp_path):
    p = save_console_image("line one\nline two", tmp_path / "console.png")
    assert p.exists() and p.stat().st_size > 0
