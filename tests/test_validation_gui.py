import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from indextts_training.dataset import _atomic_write, load_project, project_transaction
from indextts_training.manager import TRAINING_MANAGER, TrainingManager


@pytest.fixture
def gui(tmp_path):
    # Exercise the real callbacks without loading Gradio or starting the GUI.
    path = Path(__file__).resolve().parents[1] / "easy_gui.py"
    callbacks = {
        "save_dataset_project", "save_training_project", "validation_folder_value",
        "validation_project_values", "proofreading_context", "handle_proofread_event",
        "start_lora_training",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = ast.Module(body=[node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name in callbacks],
                        type_ignores=[])
    scope = dict(
        json=json, math=math, Path=Path, project_transaction=project_transaction, _atomic_write=_atomic_write,
        gr=SimpleNamespace(update=lambda **fields: fields, Error=ValueError), t=str.format,
        localize_backend_message=str, _safe_name=str, _publish_global_status=Mock(),
        _training_project_choices=Mock(return_value=[]),
        TRAINING_DIR=tmp_path / "training", TRAINING_OUTPUTS_DIR=tmp_path / "outputs",
        VALIDATION_DIR=tmp_path / "validation", MODEL_DIR=tmp_path / "model",
        LANGUAGES={"Chinese": "ZH", "English": "EN"}, NONE_SENTINEL="__none__",
    )
    exec(compile(module, str(path), "exec"), scope)
    project_file = scope["TRAINING_DIR"] / "projects/demo/project.json"
    manifest = scope["TRAINING_DIR"] / "datasets/demo/manifest.jsonl"
    _atomic_write(manifest, b'{}\n')
    _atomic_write(project_file, json.dumps({
        "project_name": "demo", "source_folder": str(tmp_path / "voice"),
        "manifest": "", "language": "ZH",
    }).encode())
    return SimpleNamespace(**scope, manifest=manifest)


def test_first_validation_folder_binding_keeps_training_settings(gui):
    suggested = str(gui.VALIDATION_DIR / "demo")
    project = load_project("demo", gui.TRAINING_DIR)
    assert gui.validation_project_values("demo", project) == (suggested, False, None)
    context = gui.proofreading_context(project["source_folder"], "demo", "English", "en_US", "validation", suggested)
    assert json.loads(context) == {
        "type": "proofread-context", "directory": "", "role": "validation", "project": "demo",
        "language": "ZH", "ui_language": "en_US",
    }
    gui.save_dataset_project("demo", "", "ignored", "ignored", "English", "validation", suggested)
    saved = load_project("demo", gui.TRAINING_DIR)
    assert Path(suggested).is_dir()
    assert gui.validation_project_values("demo", saved) == (suggested, True, None)
    assert {key: saved[key] for key in project} == project


def test_validation_workbench_event_never_updates_training_manifest(gui):
    event = json.dumps(dict(type="proofread-updated", role="validation", project="demo",
                            directory="chosen", manifest="validation.jsonl"))
    assert gui.handle_proofread_event(event, "demo", "validation") == ({}, "chosen", {}, {}, {}, {}, {}, {}, {})
    assert gui.handle_proofread_event(event, "demo", "training") == ({},) * 9


def test_rejected_switch_restores_role_and_clears_the_pending_context(gui):
    event = json.dumps(dict(type="proofread-context-rejected", role="training", project="demo",
                            directory="training-audio", language="ZH"))
    updates = gui.handle_proofread_event(event, "demo", "validation")
    assert updates[0] == "training-audio"
    assert updates[2:5] == ("training", {"value": "demo"}, "Chinese")
    assert updates[-1] == ""


def test_training_project_save_persists_switch_and_interval_with_folder(gui):
    suggested = str(gui.VALIDATION_DIR / "demo")
    gui.save_dataset_project("demo", "", "", "", "Chinese", "validation", suggested)
    gui.save_training_project(
        "demo", "", "", "Chinese", "adapter", "24 GB", "Steps", 10, 1, 3, 1,
        4, 8, 0.0, 0.001, 1, 1, 1.0, 0.0, False, 0.0, None, False, "", "", "",
        "Chinese", "", "", False, 7,
    )
    saved = load_project("demo", gui.TRAINING_DIR)
    assert gui.validation_project_values("demo", saved) == (suggested, False, 7)


def test_start_allows_no_validation_folder_only_when_switch_is_off(gui, monkeypatch):
    manager = TrainingManager()
    manager._start_locked = Mock(return_value="Training started.")
    monkeypatch.setattr(TRAINING_MANAGER, "start", manager.start)
    args = ("demo", str(gui.manifest), "adapter", "24 GB", "Steps", 1, 1, 1, 1, 4, 8, 0.0,
            0.001, 1, 1, 1.0, 0.0, False, 0.0, None, False, "", "", "", "Chinese")
    assert gui.start_lora_training(*args, False, None) == "Training started."
    assert manager._start_locked.call_args.args[0]["validation"] is None
    manager._start_locked.reset_mock()

    with pytest.raises(ValueError, match="Select and save a validation source folder"):
        gui.start_lora_training(*args, True, None)
    manager._start_locked.assert_not_called()
    assert not gui.TRAINING_OUTPUTS_DIR.exists()
