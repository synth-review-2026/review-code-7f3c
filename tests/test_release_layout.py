from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_required_paths_exist() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "configs" / "v12_core_config.json",
        ROOT / "epg_synth" / "inference.py",
        ROOT / "epg_synth" / "model_factory.py",
        ROOT / "DWT_IDWT" / "DWT_IDWT_layer.py",
        ROOT / "guided_diffusion" / "enhancement_predictor.py",
        ROOT / "scripts" / "evaluate_image_roi_118.py",
        ROOT / "scripts" / "smoke_test.py",
        ROOT / "scripts" / "make_sha256sums.py",
    ]
    assert all(path.exists() for path in required)


def test_no_clinical_volumes_are_packaged() -> None:
    forbidden = ("*.nii", "*.nii.gz", "*.dcm", "*.nrrd")
    assert not [
        path
        for pattern in forbidden
        for path in ROOT.rglob(pattern)
    ]


def test_no_checkpoint_binaries_are_packaged() -> None:
    forbidden = ("*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.onnx")
    assert not [
        path
        for pattern in forbidden
        for path in ROOT.rglob(pattern)
    ]


def test_no_local_absolute_paths_remain() -> None:
    forbidden = (
        "C:" + "\\" + "Users" + "\\",
        "D:" + "\\" + "EPG-Synth",
        "vip" + "user",
    )
    matches = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".md",
            ".json",
            ".toml",
            ".txt",
            ".gitignore",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in forbidden):
            matches.append(str(path.relative_to(ROOT)))
    assert not matches


def test_no_cache_files_are_packaged() -> None:
    assert not list(ROOT.rglob("__pycache__"))
    assert not list(ROOT.rglob("*.pyc"))


def test_release_manifest_is_consistent() -> None:
    freeze = json.loads(
        (ROOT / "configs" / "final_model_freeze.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (ROOT / "configs" / "v12_core_config.json").read_text(encoding="utf-8")
    )
    assert freeze["model_name"] == config["model_name"] == "V12-Balanced"
    assert freeze["parameter_count"] == 188352266
    assert freeze["inference_module"] == "epg_synth.inference"
    assert set(freeze["checkpoints"]) == {
        "old_enhancer",
        "v12_enhancer",
        "base_direct",
        "v12_finetuned",
    }
    assert set(config["checkpoints"]) == set(freeze["checkpoints"])
