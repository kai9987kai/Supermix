"""Static browser contract tests for the fail-closed NexusMind Studio."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STUDIO = ROOT / "web_static" / "nexus_studio.html"


def test_studio_uses_api_request_schema_and_has_no_fabricated_local_results():
    text = STUDIO.read_text(encoding="utf-8")

    assert "fetch('/v1/solve'" in text
    assert "JSON.stringify({query:q})" in text
    assert "fetch('/v1/innovate'" in text
    assert "JSON.stringify({topic:q,count:count})" in text
    assert "fetch('/v1/swarm'" in text
    assert "JSON.stringify({query:q,max_rounds:maxRounds})" in text
    assert "fetch('/v1/got'" in text
    assert "JSON.stringify({query:q,max_depth:depth})" in text

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


def test_studio_fails_closed_and_requires_machine_readable_evidence_contracts():
    text = STUDIO.read_text(encoding="utf-8")

    assert "DEMO / BACKEND UNAVAILABLE" in text
    assert "No result was generated locally" in text
    assert "function analysisContractValid" in text
    assert "function verifiedContractValid" in text
    assert "function classifyThinkResponse" in text
    assert "No backend candidate was displayed" in text
    assert "browser withheld the backend candidate and trace" in text
    assert "verifier.id === 'grounding_runtime.finalize_grounded_response'" in text
    assert "verifier.independent_recompute === true" in text
    assert "result.epistemics.decision === 'analysis_only'" in text
    assert "decision !== 'answered'" in text
    assert "result.answer_authority !== true" in text
    assert "Receipts are audit metadata, not authority" in text
    assert "function esc(value)" in text
    assert "esc(text).replace" in text


def test_studio_ignores_stale_solver_and_think_completions():
    text = STUDIO.read_text(encoding="utf-8")

    assert "requests: { solver:0, think:0 }" in text
    assert "var requestToken = ++state.requests.solver" in text
    assert "requestToken !== state.requests.solver" in text
    assert "var requestToken = ++state.requests.think" in text
    assert "requestToken !== state.requests.think" in text
    assert "Verified for submitted request:" in text


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
  'verifiedContractValid',
  'nonAuthoritativeVerifierValid',
  'thinkAnalysisContractValid',
  'thinkAbstentionContractValid',
  'classifyThinkResponse'
].map(extract).join('\n'));
const none = {id:'none', passed:false, independent_recompute:false};
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
const got = [classifyThinkResponse(contradictory), classifyThinkResponse(validAbstention)];
if (JSON.stringify(got) !== JSON.stringify(['invalid', 'abstained'])) {
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
