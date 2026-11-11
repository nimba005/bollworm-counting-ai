import pytest

from bollworm.predict import counts_at, tune_thresholds
from bollworm.utils import competition_mae, image_stem, row_mae, split_image_class_id, wkt_to_xyxy


@pytest.mark.parametrize(
    "rid, expected",
    [
        ("ID_D9ONL553_ABW", ("ID_D9ONL553", "abw")),
        ("id_0002ea6f15c7_pbw", ("id_0002ea6f15c7", "pbw")),
        ("id_abc.jpg_pbw", ("id_abc", "pbw")),
        ("id_abc_abw.jpg", ("id_abc", "abw")),
    ],
)
def test_split_id(rid, expected):
    assert split_image_class_id(rid) == expected


def test_split_id_bad():
    with pytest.raises(ValueError):
        split_image_class_id("id_abc")


def test_image_stem():
    assert image_stem("id_x.JPG") == "id_x"
    assert image_stem("some/dir/id_x.jpeg") == "id_x"
    assert image_stem("id_x") == "id_x"


def test_wkt():
    assert wkt_to_xyxy("POLYGON ((10 20, 30 20, 30 40, 10 40, 10 20))") == (10, 20, 30, 40)
    assert wkt_to_xyxy("MULTIPOLYGON (((1.5 2, 3 2, 3 4, 1.5 4, 1.5 2)))") == (1.5, 2, 3, 4)
    assert wkt_to_xyxy(float("nan")) is None
    assert wkt_to_xyxy("POLYGON EMPTY") is None


def test_metric():
    gt = {"a": (3, 1), "b": (0, 0)}
    pred = {"a": (1, 2), "b": (0, 1)}
    assert competition_mae(gt, pred) == pytest.approx((3 + 1) / 2)
    assert row_mae(gt, pred) == pytest.approx(1.0)


def test_threshold_tuning():
    dets = {"a": [[0.9, 0.8, 0.3], [0.7]], "b": [[0.2], []]}
    gt = {"a": (2, 1), "b": (0, 0)}
    thr, mae = tune_thresholds(dets, gt)
    assert mae == 0
    assert counts_at(dets, thr) == gt
