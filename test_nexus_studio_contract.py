"""Static browser contract tests for the fail-closed NexusMind Studio."""

from __future__ import annotations

import subprocess
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STUDIO = ROOT / "web_static" / "nexus_studio.html"


class _PanelNestingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.nested_panels = []

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            return
        attr_map = dict(attrs)
        node_id = attr_map.get("id", "")
        if node_id.startswith("panel-"):
            parent = next(
                (value for value in reversed(self.stack) if value.startswith("panel-")),
                None,
            )
            if parent is not None:
                self.nested_panels.append((node_id, parent))
        self.stack.append(node_id)

    def handle_endtag(self, tag):
        if tag == "div" and self.stack:
            self.stack.pop()


def test_studio_uses_api_request_schema_and_has_no_fabricated_local_results():
    text = STUDIO.read_text(encoding="utf-8")

    assert "fetch('/v1/solve'" in text
    assert "query:q,request_nonce:requestNonce" in text
    assert "fetch('/v1/verify'" in text
    assert "proof_capsule:capsule" in text
    assert "fetch('/v1/innovate'" in text
    assert "JSON.stringify({topic:q,count:count})" in text
    assert "fetch('/v1/swarm'" in text
    assert "JSON.stringify({query:q,max_rounds:maxRounds})" in text
    assert "fetch('/v1/got'" in text
    assert "JSON.stringify({query:q,max_depth:depth})" in text
    assert "fetch('/v1/risk-control'" in text
    assert "fetch('/v1/risk-control/audit'" in text
    assert "function runRiskAudit" in text

    for forbidden in (
        "function localSolve",
        "function localIdeate",
        "function localSwarm",
        "function localGoT",
        "function localChat",
        "Math.random",
        "40-60% improvement",
        "Verified consensus conclusion",
        "Solver Receipts',fmt:function(){return '100%'",
    ):
        assert forbidden not in text


def test_workspace_tab_panels_are_siblings_not_nested_inside_other_tabs():
    text = STUDIO.read_text(encoding="utf-8")
    parser = _PanelNestingParser()
    parser.feed(text)

    assert parser.nested_panels == []
    assert "#panel-telem.active{display:grid;}" in text
    assert "#panel-telem{padding:20px;overflow-y:auto;display:grid" not in text


def test_studio_fails_closed_and_requires_machine_readable_evidence_contracts():
    text = STUDIO.read_text(encoding="utf-8")

    assert "DEMO / BACKEND UNAVAILABLE" in text
    assert "No result was generated locally" in text
    assert "function analysisContractValid" in text
    assert "function verifiedPublicOutput" in text
    assert "function verifiedRenderedAnswer" in text
    assert "function verifiedContractValid" in text
    assert "function classifyThinkResponse" in text
    assert "No backend candidate was displayed" in text
    assert "browser withheld the backend candidate and trace" in text
    assert "verifier.id === 'grounding_runtime.finalize_grounded_response'" in text
    assert "verifier.fresh_recompute === true" in text
    assert "verifier.algorithmically_independent === false" in text
    assert "nexus-proof-carrying-number-v2" in text
    assert "nexus-independent-arithmetic-checker-v1" in text
    assert "nexus-independent-science-checker-v1" in text
    assert "function rendererRevalidate" in text
    assert "epistemics.decision === 'analysis_only'" in text
    assert "epistemics.answer_authority === false" in text
    assert "nonAuthoritativeVerifierValid(epistemics)" in text
    assert "decision !== 'answered'" in text
    assert "result.answer_authority !== true" in text
    assert "Capsules and receipts are audit metadata" in text
    assert "function esc(value)" in text
    assert "esc(text).replace" in text
    assert "appendMsg('ai', verifiedRenderedAnswer(result)" in text
    assert "notice.textContent = result.output" not in text
    assert "no local risk result was generated" in text
    assert "shadow receipt only" in text


def test_verified_renderers_use_only_capsule_bound_response_fields():
    text = STUDIO.read_text(encoding="utf-8")
    solver_renderer = text.split("function renderVerifiedSolverResult", 1)[1].split(
        "/* ─── IDEATION", 1
    )[0]
    think_renderer = text.split("function renderThinkResultAfterVerification", 1)[1].split(
        "/* ─── INIT", 1
    )[0]

    assert "capsuleResult.display_answer" in solver_renderer
    assert "capsuleResult.unit" in solver_renderer
    assert "capsuleResult.problem_class" in solver_renderer
    assert "capsuleResult.method" in solver_renderer
    for unbound in (
        "result.display_answer",
        "result.unit",
        "result.domain",
        "result.formula_id",
        "result.target",
        "result.steps",
        "result.receipt",
    ):
        assert unbound not in solver_renderer
    assert "var capsuleResult = result.proof_capsule.result || {}" in text
    assert "String(capsuleResult.unit || '')" in text
    assert "if (!verified && !abstained && res.thought_steps" in think_renderer
    assert "verified ? verifiedRenderedAnswer(res)" in think_renderer
    assert "bindings.surface === expectedSurface" in text
    assert "capsuleBindings.surface === expectedSurface" in text
    assert "surface:expectedSurface" in text


def test_studio_ignores_stale_chat_solver_and_think_completions():
    text = STUDIO.read_text(encoding="utf-8")

    assert "requests: { chat:0, solver:0, think:0 }" in text
    assert "var requestToken = ++state.requests.chat" in text
    assert "requestToken !== state.requests.chat" in text
    assert "var requestToken = ++state.requests.solver" in text
    assert "requestToken !== state.requests.solver" in text
    assert "var requestToken = ++state.requests.think" in text
    assert "requestToken !== state.requests.think" in text
    assert "Verified result:" in text


def test_studio_classifies_contradictory_think_response_as_invalid():
    script = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
function extract(name) {
  const start = html.indexOf('function ' + name + '(');
  if (start < 0) throw new Error('missing function ' + name);
  const open = html.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < html.length; i++) {
    if (html[i] === '{') depth++;
    if (html[i] === '}') {
      depth--;
      if (depth === 0) return html.slice(start, i + 1);
    }
  }
  throw new Error('unterminated function ' + name);
}
eval([
  'verifiedPublicOutput',
  'verifiedRenderedAnswer',
  'verifiedContractValid',
  'nonAuthoritativeVerifierValid',
  'thinkAnalysisContractValid',
  'thinkAbstentionContractValid',
  'classifyThinkResponse'
].map(extract).join('\n'));
const none = {id:'none', passed:false, fresh_recompute:false, algorithmically_independent:false};
const contradictory = {
  confidence:0.9,
  output:'Answer 15',
  epistemics:{decision:'abstained', answer_authority:false,
    evidence_class:'no_applicable_verifier', verifier:none}
};
const validAbstention = {
  confidence:null,
  output:'untrusted candidate',
  epistemics:{decision:'abstained', answer_authority:false,
    evidence_class:'no_applicable_verifier', verifier:none}
};
const mismatchedAliases = {output:'The exact result is 14.', reply:'The exact result is 999.'};
const got = [
  classifyThinkResponse(contradictory),
  classifyThinkResponse(validAbstention),
  verifiedPublicOutput(mismatchedAliases)
];
if (JSON.stringify(got) !== JSON.stringify(['invalid', 'abstained', ''])) {
  throw new Error('unexpected classifications: ' + JSON.stringify(got));
}
"""
    result = subprocess.run(
        ["node", "-e", script, str(STUDIO)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
