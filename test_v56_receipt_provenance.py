"""Guard the link between what the documents claim and which checkpoint produced it.

This repository's central promise is that every number is measured and every
measurement is reproducible from a named command. That promise broke silently
once already: `README.md` and two module docstrings told the reader to run the
v56 promotion gate, the reasoner web app and the chat benchmark against
`output/v56_curriculum/v56_curriculum_160k.pt`, while every v56 number printed
around those commands came from `output/v56b_randslots_entropy/`. The first
checkpoint scores **0.9220** and gates at 0.9329; the quoted figures are
**0.9740** and **0.9762**. Following the README exactly reproduced the
superseded run and nothing said so.

Nothing detected it because no test compares prose to provenance. These do.
They are deliberately cheap -- they read JSON receipts and grep markdown, never
loading a model -- so they can run on every change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
V56_DOC = ROOT / "docs" / "V56_LATENT_STATE_REASONER.md"
GATE_RECEIPT = ROOT / "output" / "v56_promotion_gate.json"
CHAT_RECEIPT = ROOT / "output" / "v56_chat_benchmark.json"

#: Any `output/<run>/<file>.pt` mentioned in prose or a usage example.
CHECKPOINT_REFERENCE = re.compile(r"output[/\\][\w.\-]+[/\\][\w.\-]+\.(?:pt|pth)")

#: The checkpoint *under test* in a command, as opposed to `--baseline`, which
#: names the v51 model the candidate is measured against and must stay different.
EVALUATED_CHECKPOINT = re.compile(
    r"--(?:candidate|checkpoint)[=\s]+(output[/\\][\w.\-]+[/\\][\w.\-]+\.(?:pt|pth))"
)

V56_COMMAND_SOURCES = (
    README,
    ROOT / "source" / "run_v56_promotion_gate.py",
    ROOT / "source" / "benchmark_reasoner_chat.py",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def promoted_checkpoint() -> str:
    """The checkpoint the promotion gate actually scored, from its own receipt."""

    receipt = json.loads(read(GATE_RECEIPT))
    return Path(receipt["candidate"]["checkpoint"]).as_posix()


@pytest.mark.skipif(not GATE_RECEIPT.exists(), reason="v56 promotion gate receipt not present")
def test_the_gate_receipt_names_a_checkpoint_that_exists():
    assert (ROOT / promoted_checkpoint()).exists(), (
        f"the promotion gate receipt names {promoted_checkpoint()}, which is not on disk"
    )


@pytest.mark.skipif(not GATE_RECEIPT.exists(), reason="v56 promotion gate receipt not present")
def test_v56_reproduction_commands_name_the_promoted_checkpoint():
    """The exact failure this file exists for.

    Every v56 checkpoint path in a runnable command must be the one the gate
    promoted. A command pointing at a superseded run reproduces a number the
    surrounding prose does not claim.
    """

    promoted = promoted_checkpoint()
    wrong: list = []
    checked = 0
    for path in V56_COMMAND_SOURCES:
        if not path.exists():
            continue
        for line in read(path).splitlines():
            for reference in EVALUATED_CHECKPOINT.findall(line):
                normalised = reference.replace("\\", "/")
                # Only v56 commands are in scope; other lines evaluate other lines
                # of work against their own checkpoints.
                if "v56" not in normalised:
                    continue
                checked += 1
                if normalised != promoted:
                    wrong.append(f"{path.name}: {normalised}")

    assert checked, "no v56 --candidate/--checkpoint command was found to check"
    assert not wrong, (
        "these v56 commands evaluate a checkpoint other than the promoted "
        f"{promoted}:\n  " + "\n  ".join(wrong)
    )


@pytest.mark.skipif(not CHAT_RECEIPT.exists(), reason="v56 chat benchmark receipt not present")
def test_the_chat_benchmark_table_and_its_command_use_one_checkpoint():
    """`README.md` prints the chat benchmark table straight from its receipt, so
    the command printed above the table has to be the command that produced it."""

    receipt = json.loads(read(CHAT_RECEIPT))
    used = Path(receipt["checkpoint"]).as_posix()
    readme = read(README)
    command_lines = [
        line for line in readme.splitlines() if "benchmark_reasoner_chat.py" in line
    ]
    assert command_lines, "README no longer documents the chat benchmark command"
    for line in command_lines:
        for reference in CHECKPOINT_REFERENCE.findall(line):
            assert reference.replace("\\", "/") == used, (
                f"README runs the chat benchmark against {reference}, but the table "
                f"it prints was measured on {used}"
            )


@pytest.mark.skipif(not V56_DOC.exists(), reason="v56 document not present")
def test_the_v56_document_lists_the_promoted_checkpoints_receipt():
    """A document that cites receipts must cite the one behind its headline.

    The v56 document's `Receipts:` list named four files, none of which was the
    receipt for the promoted checkpoint whose numbers the document reports.
    """

    promoted_dir = Path(promoted_checkpoint()).parent.name
    assert f"output/{promoted_dir}/benchmark_results.json" in read(V56_DOC), (
        f"docs/V56_LATENT_STATE_REASONER.md never cites output/{promoted_dir}/"
        "benchmark_results.json, the receipt behind its headline numbers"
    )


@pytest.mark.skipif(not GATE_RECEIPT.exists(), reason="v56 promotion gate receipt not present")
def test_the_documented_accuracy_matches_the_promoted_checkpoints_receipt():
    """The headline accuracy in both documents must be the promoted run's own.

    Reads the number out of the receipt rather than hard-coding it, so a
    re-measured model updates this test by being re-measured.
    """

    promoted_dir = Path(promoted_checkpoint()).parent
    receipt_path = ROOT / promoted_dir / "benchmark_results.json"
    if not receipt_path.exists():
        pytest.skip(f"no benchmark receipt at {receipt_path}")
    accuracy = json.loads(read(receipt_path))["eval_default"]["accuracy"]
    rendered = f"{accuracy:.4f}"

    for document in (README, V56_DOC):
        if not document.exists():
            continue
        assert rendered in read(document), (
            f"{document.name} never states the promoted checkpoint's held-out "
            f"accuracy {rendered}"
        )


@pytest.mark.skipif(not GATE_RECEIPT.exists(), reason="v56 promotion gate receipt not present")
def test_the_superseded_checkpoint_is_labelled_where_it_is_still_mentioned():
    """Keeping the earlier run in the ablation table is correct; leaving it
    unlabelled is what made it look like the headline."""

    doc = read(V56_DOC)
    if "output/v56_curriculum/" not in doc:
        pytest.skip("the superseded run is no longer mentioned")
    assert "0.9220" in doc, (
        "docs/V56_LATENT_STATE_REASONER.md still cites output/v56_curriculum/ but no "
        "longer states its 0.9220 accuracy, so a reader cannot tell it is the earlier run"
    )
