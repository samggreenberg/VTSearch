"""imagekit.util: the image-tensor convention, letterboxing, NMS, transforms, CLI input selection."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from imagekit import decode_image_buffer, read_image_tensor
from imagekit.util import (
    Sample,
    apply_transforms,
    build_transforms,
    clamp_box,
    collect_image_paths,
    ensure_image_tensor,
    letterbox,
    load_model_config,
    nms,
    resize_within,
    resolve_model_file,
)


def _png_bytes(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


class TestImageTensor:
    def test_rgb_uint8_passes_through_contiguous(self):
        arr = np.zeros((4, 6, 3), np.uint8)[:, ::-1]
        out = ensure_image_tensor(arr)
        assert out.shape == (4, 6, 3) and out.flags.c_contiguous

    def test_gray_and_rgba_are_coerced(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        out = ensure_image_tensor(gray)
        assert out.shape == (3, 4, 3)
        assert (out[..., 0] == gray).all() and (out[..., 2] == gray).all()
        rgba = np.zeros((2, 2, 4), np.uint8)
        assert ensure_image_tensor(rgba).shape == (2, 2, 3)

    @pytest.mark.parametrize("bad", [np.zeros((2, 2, 2), np.uint8), np.zeros((0, 3, 3), np.uint8), np.zeros((2, 2, 3))])
    def test_wrong_shape_or_dtype_is_refused(self, bad):
        with pytest.raises(ValueError):
            ensure_image_tensor(bad)

    def test_decode_round_trips_rgb(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (5, 7, 3), dtype=np.uint8)
        assert (decode_image_buffer(_png_bytes(arr)) == arr).all()

    def test_decode_refuses_garbage_and_empty(self):
        with pytest.raises(ValueError):
            decode_image_buffer(b"not an image")
        with pytest.raises(ValueError):
            decode_image_buffer(b"")

    def test_read_image_tensor(self, tmp_path):
        arr = np.full((3, 3, 3), 7, np.uint8)
        (tmp_path / "a.png").write_bytes(_png_bytes(arr))
        assert (read_image_tensor(tmp_path / "a.png") == arr).all()
        with pytest.raises(FileNotFoundError):
            read_image_tensor(tmp_path / "missing.png")


class TestGeometry:
    def test_letterbox_pads_bottom_right_and_reports_scale(self):
        img = np.full((100, 200, 3), 255, np.uint8)
        out, scale = letterbox(img, 64, 64)
        assert out.shape == (64, 64, 3) and scale == pytest.approx(0.32)
        assert out[:32, :].min() == 255  # the image, 32 rows tall
        assert out[32:, :].max() == 0  # the padding below it

    def test_resize_within_only_shrinks(self):
        img = np.zeros((10, 20, 3), np.uint8)
        same, s1 = resize_within(img, 50)
        assert same is img and s1 == 1.0
        small, s2 = resize_within(img, 10)
        assert small.shape == (5, 10, 3) and s2 == 0.5

    def test_nms_keeps_the_best_of_overlapping_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
        scores = np.array([0.5, 0.9, 0.7], np.float32)
        assert nms(boxes, scores, 0.45).tolist() == [1, 2]
        assert nms(np.zeros((0, 4)), np.zeros((0,)), 0.45).size == 0

    def test_clamp_box(self):
        assert clamp_box((-5, -5, 50, 50), 20, 10) == (0.0, 0.0, 20.0, 10.0)


class TestTransforms:
    def test_normalize_both_spellings_agree(self):
        img = np.full((2, 2, 3), 200, np.uint8)
        a = apply_transforms(
            build_transforms(
                [{"name": "Normalize", "args": {"mean_vals": [127.5] * 3, "scale_vals": [1 / 127.5] * 3}}]
            ),
            Sample(img),
        )
        b = apply_transforms(
            build_transforms([{"name": "Normalize", "args": {"mean": [0.5] * 3, "std": [0.5] * 3}}]), Sample(img)
        )
        np.testing.assert_allclose(a.image, b.image, atol=1e-5)
        assert a.image.dtype == np.float32

    def test_resize_letterbox_and_to_tensor(self):
        img = np.zeros((50, 100, 3), np.uint8)
        chain = build_transforms(
            [
                {"name": "Resize", "args": {"height": 64, "width": 64, "preserve_aspect": True, "pad": 1}},
                {"name": "ToTensor"},
            ]
        )
        out = apply_transforms(chain, Sample(img))
        assert out.tensor is not None and out.tensor.shape == (1, 3, 64, 64)
        assert out.scale == pytest.approx(0.64)

    def test_resize_max_size_and_stretch(self):
        img = np.zeros((50, 100, 3), np.uint8)
        capped = apply_transforms(build_transforms([{"name": "Resize", "args": {"max_size": 50}}]), Sample(img))
        assert capped.image.shape == (25, 50, 3) and capped.scale == 0.5
        stretched = apply_transforms(
            build_transforms([{"name": "Resize", "args": {"height": 10, "width": 10, "preserve_aspect": False}}]),
            Sample(img),
        )
        assert stretched.image.shape == (10, 10, 3) and stretched.scale == 1.0

    def test_channel_swap_reverses_the_channels(self):
        img = np.zeros((1, 1, 3), np.uint8)
        img[0, 0] = (1, 2, 3)
        out = apply_transforms(build_transforms([{"name": "ChannelSwap"}]), Sample(img))
        assert out.image[0, 0].tolist() == [3, 2, 1] and out.image.flags.c_contiguous

    def test_unknown_transform_is_named(self):
        with pytest.raises(KeyError, match="Nope"):
            build_transforms([{"name": "Nope"}])


class TestModelFiles:
    def test_resolve_model_file_accepts_file_or_single_model_dir(self, tmp_path):
        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        assert resolve_model_file(model) == model
        assert resolve_model_file(tmp_path) == model
        (tmp_path / "other.onnx").write_bytes(b"y")
        with pytest.raises(FileNotFoundError, match="exactly one"):
            resolve_model_file(tmp_path)
        with pytest.raises(FileNotFoundError):
            resolve_model_file(tmp_path / "nothing")

    def test_load_model_config_lookup_order(self, tmp_path):
        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        default = tmp_path / "default.json"
        default.write_text(json.dumps({"model": {"name": "default"}}))
        assert load_model_config(model, default_path=default)["model"]["name"] == "default"
        sidecar = tmp_path / "m.json"
        sidecar.write_text(json.dumps({"model": {"name": "sidecar"}}))
        assert load_model_config(model, default_path=default)["model"]["name"] == "sidecar"
        explicit = tmp_path / "x.json"
        explicit.write_text(json.dumps({"model": {"name": "explicit"}}))
        assert load_model_config(model, explicit)["model"]["name"] == "explicit"
        with pytest.raises(FileNotFoundError, match="looked at"):
            load_model_config(model, tmp_path / "absent.json")
        with pytest.raises(FileNotFoundError):
            load_model_config(tmp_path / "lonely.onnx")


class TestCollectImagePaths:
    def test_exactly_one_selector(self, tmp_path):
        with pytest.raises(ValueError):
            collect_image_paths()
        with pytest.raises(ValueError):
            collect_image_paths(directory=tmp_path, image=tmp_path / "a.png")

    def test_dir_walks_images_only_sorted(self, tmp_path):
        (tmp_path / "sub").mkdir()
        for name in ("b.jpg", "a.PNG", "notes.txt", "sub/c.webp"):
            (tmp_path / name).write_bytes(b"")
        found = [p.relative_to(tmp_path).as_posix() for p in collect_image_paths(directory=tmp_path)]
        assert found == ["a.PNG", "b.jpg", "sub/c.webp"]
        with pytest.raises(NotADirectoryError):
            collect_image_paths(directory=tmp_path / "nope")

    def test_list_skips_blank_and_comment_lines(self, tmp_path):
        listing = tmp_path / "list.txt"
        listing.write_text("# header\n\n/x/a.png\n  /x/b.png  \n")
        assert [str(p) for p in collect_image_paths(list_file=listing)] == ["/x/a.png", "/x/b.png"]

    def test_img_is_one_path(self, tmp_path):
        assert collect_image_paths(image=tmp_path / "z.png") == [tmp_path / "z.png"]
