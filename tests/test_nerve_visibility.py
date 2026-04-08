from pathlib import Path

from vlm_navigator.utils.nerve_visibility import (
    VisibilityThresholds,
    build_static_mask,
    classify_visibility,
    step_images_in_folder,
    visibility_score,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _episode_folder(name: str) -> Path:
    return _repo_root() / "results" / "manual_test" / name


def test_visibility_pipeline_on_recorded_episode():
    """
    Smoke test on recorded screenshots.

    This test is intentionally lightweight: it verifies that visibility scoring
    runs end-to-end and yields a mix of classes for manual inspection.
    """
    folder = _episode_folder("gpt-5_pos1_present_trial01")
    if not folder.exists():
        # Keep tests usable on machines without local screenshots.
        return

    images = step_images_in_folder(folder)
    assert len(images) > 0, "No step images found for recorded episode."

    static_mask = build_static_mask(images, freq_threshold=0.9)
    thresholds = VisibilityThresholds(not_visible_max=0.0012, visible_min=0.0030)

    labels = []
    for image_path in images:
        score = visibility_score(image_path, static_mask=static_mask)
        label = classify_visibility(score["dynamic_green_fraction"], thresholds=thresholds)
        labels.append(label)

    # We expect this run to include both visible and not_visible periods.
    assert "visible" in labels
    assert "not_visible" in labels


def test_visibility_score_fields_present():
    folder = _episode_folder("gpt-5_pos1")
    if not folder.exists():
        return

    images = step_images_in_folder(folder)
    assert images, "No step images found."

    static_mask = build_static_mask(images, freq_threshold=0.9)
    score = visibility_score(images[0], static_mask=static_mask)
    expected = {
        "total_green_fraction",
        "dynamic_green_fraction",
        "total_green_pixels",
        "dynamic_green_pixels",
        "left_panel_pixels",
    }
    assert expected.issubset(score.keys())
