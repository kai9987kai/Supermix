"""Deterministic Plan-Evaluate interaction intelligence for Supermix chat.

The planner converts observable request cues into a cautious interaction plan:
intent, ambiguity, epistemic risk, response obligations, and bounded ranking
weights.  It then audits the selected response and applies only narrow repairs.

This module is deliberately model-independent and inspectable.  It does not
claim to infer a user's internal emotional state, and its compute advice is
shadow-only: the checkpoint-bound v51 decision verifier remains the only
authority for adaptive model exits.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PLANNER_VERSION = "supermix-plan-evaluate-v4"

_EMOTIONAL_RE = re.compile(
    r"\b(feel|feeling|sad|anxious|scared|afraid|upset|angry|lonely|hurt|grief|"
    r"overwhelm(?:ed)?|stress(?:ed)?|worried|frustrated|devastated|hopeless)\b",
    re.I,
)
_POSITIVE_RE = re.compile(
    r"\b(happy|excited|hopeful|relieved|grateful|proud|love|great news)\b",
    re.I,
)
_HIGH_AROUSAL_RE = re.compile(
    r"\b(urgent|immediately|panic|terrified|furious|desperate|right now|asap)\b|!{2,}",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(help me cope|need support|comfort me|listen to me|what should i do|"
    r"can't handle|cannot handle)\b",
    re.I,
)
_PROBLEM_RE = re.compile(
    r"\b(solve|fix|debug|diagnose|plan|implement|build|optimi[sz]e|improve|"
    r"calculate|derive|prove|root cause|trade-?off|step by step)\b",
    re.I,
)
_DECISION_RE = re.compile(
    r"\b(choose|decide|which option|compare|recommend|best choice|pros and cons|trade-?off)\b",
    re.I,
)
_EXPLAIN_RE = re.compile(
    r"\b(explain|why|how does|teach|understand|walk me through|what is|what are)\b",
    re.I,
)
_CREATIVE_RE = re.compile(
    r"\b(create|write|brainstorm|imagine|invent|story|poem|novel|creative|ideas?)\b",
    re.I,
)
_EDIT_RE = re.compile(
    r"\b(rewrite|rephrase|edit|polish|shorten|expand|refine|proofread)\b",
    re.I,
)
_LOOKUP_RE = re.compile(
    r"\b(exact|fact|factual|source|citation|cite|evidence|statistics?|date|who is|"
    r"current|latest|today|verbatim|definition|how many|price|law|policy)\b",
    re.I,
)
_HIGH_STAKES_RE = re.compile(
    r"\b(medical|diagnosis|symptom|dose|medication|legal|lawyer|court|financial|"
    r"investment|tax|suicid(?:e|al|ality)|self harm|kill myself|emergency)\b",
    re.I,
)
_PERSONAL_CRISIS_RE = re.compile(
    r"\b(?:kill(?:ing)?|hurt(?:ing)?)\s+myself\b"
    r"(?!\s+(?:laughing\b|with\s+laughter\b))|"
    r"\bend\s+my\s+life(?![\s-]+insurance\b)\b|"
    r"\bi(?:'m| am| feel)\s+(?:very\s+)?suicidal\b|"
    r"\bi\s+(?:might|may|want|plan|intend|am going)\s+(?:to\s+)?"
    r"(?:commit suicide|self[- ]?harm|hurt myself|kill myself|"
    r"end my life(?![\s-]+insurance\b))\b|"
    r"\bi(?:'m| am|'ve been| have been| keep)\s+thinking\s+about\s+"
    r"(?:suicide|self[- ]?harm|killing myself|ending my life)\b|"
    r"\bhow\s+(?:do|can|could|should|would)\s+i\s+"
    r"(?:best\s+)?commit\s+suicide\b|"
    r"\bwhat(?:'s| is| are)\s+(?:the\s+)?(?:best\s+)?"
    r"(?:ways?|methods?)\s+(?:(?:for\s+me\s+)?to\s+)?"
    r"commit\s+suicide\b|"
    r"\b(?:the\s+)?best\s+way\s+to\s+commit\s+suicide\b|"
    r"\bmy\s+(?:suicidal|self[- ]?harm)\s+(?:thoughts|urges|plans?)\b|"
    r"\b(?:might act on it|can't stay safe|cannot stay safe)\b",
    re.I,
)
_PERSONAL_URGENT_HEALTH_RE = re.compile(
    r"\bi(?:'m| am)\s+(?:having|experiencing)\s+(?:(?:sudden|severe|new|"
    r"bad|intense)\s+)?(?:chest pain|shortness of breath|difficulty breathing|"
    r"severe bleeding|anaphylaxis|signs? of (?:a )?stroke)\b|"
    r"\bi\s+have\s+(?:(?:sudden|severe|new|bad|intense)\s+)?"
    r"(?:chest pain|shortness of breath|difficulty breathing|severe bleeding|"
    r"anaphylaxis|signs? of (?:a )?stroke)\b|"
    r"\bi(?:\s+am|'m)?\s*(?:overdosing|unconscious)\b|"
    r"\bi\s+(?:think\s+i\s+)?(?:have\s+)?overdosed\b|"
    r"\bi\s+(?:can't|cannot)\s+breathe\b|"
    r"\bmy\s+(?:friend|partner|child|parent)\s+(?:is|has|just)\s+"
    r"(?:unconscious|overdosed|overdosing|severe bleeding|anaphylaxis|"
    r"chest pain|difficulty breathing|signs? of (?:a )?stroke)\b|"
    r"\bsomeone\s+(?:is|has|just)\s+(?:unconscious|overdosed|overdosing|"
    r"severe bleeding|anaphylaxis|chest pain|difficulty breathing|"
    r"signs? of (?:a )?stroke)\b|"
    r"\b(?:chest pain|shortness of breath|difficulty breathing|severe bleeding|"
    r"overdosing|anaphylaxis|signs? of (?:a )?stroke)\b.{0,32}"
    r"\b(?:right now|currently|just started)\b",
    re.I,
)
_REPORTED_SAFETY_CONTEXT_RE = re.compile(
    r"\b(historical|fictional|hypothetical|quoted|quotation|example|character|"
    r"phrase|term|wording|sentence)\b",
    re.I,
)
_NONCRISIS_IDIOM_RE = re.compile(
    r"\blife[- ]insurance\b|"
    r"\b(?:was\s+|had\s+me\s+)?killing\s+myself\s+laughing\b",
    re.I,
)
_CURRENT_DISCLOSURE_OVERRIDE_RE = re.compile(
    r"\b(?:but|however|actually|in fact|right now|currently)\b.{0,96}"
    r"(?:kill(?:ing)? myself|hurt(?:ing)? myself|"
    r"end my life(?![\s-]+insurance\b)|"
    r"i(?:'m| am| feel) suicidal|can't stay safe|cannot stay safe|"
    r"chest pain|shortness of breath|difficulty breathing|can't breathe|"
    r"cannot breathe|severe bleeding|overdos(?:e|ed|ing)|anaphylaxis|"
    r"signs? of (?:a )?stroke)",
    re.I,
)
_QUOTED_SPAN_RE = re.compile(
    "\"[^\"\\n]{0,500}\"|“[^”\\n]{0,500}”|"
    "(?<!\\w)'[^'\\n]{0,500}'(?!\\w)|‘[^’\\n]{0,500}’"
)
_RECENCY_RE = re.compile(
    r"\b(current|latest|today|now|recent|newest|this (?:week|month|year))\b",
    re.I,
)
_AGREEMENT_REQUEST_RE = re.compile(
    r"\b(tell me (?:i(?:'m| am)|we(?:'re| are)) right|agree with me|"
    r"validate my claim|back me up|say yes|don't disagree|am i right)\b",
    re.I,
)
_CERTAINTY_FRAMING_RE = re.compile(
    r"\b(i know|i am certain|i'm certain|obviously|definitely|must be true|"
    r"proves? that|without a doubt)\b",
    re.I,
)
_DEICTIC_RE = re.compile(
    r"\b(this|that|it|same|again|do that|make it better)\b",
    re.I,
)
_AFFECT_CONTINUITY_RE = re.compile(
    r"\b(still|what next|what should i do|how do i handle|now what|keep going)\b",
    re.I,
)
_MULTISTEP_RE = re.compile(
    r"\b(step by step|multi-?step|first .+ then|plan|debug|diagnose|derive|prove|"
    r"root cause|trade-?off|compare|evaluate|test|verify)\b",
    re.I,
)
_CONSTRAINT_RE = re.compile(
    r"\b(must|should|without|while|but also|at least|at most|exactly|only|"
    r"require(?:ment|d)?|constraint|format|deadline|budget)\b",
    re.I,
)
_CODE_MATH_RE = re.compile(
    r"\b(traceback|exception|stack trace|function|class|algorithm|equation|"
    r"calculate|derivative|integral|proof|complexity|benchmark|test suite)\b|```",
    re.I,
)
_COMPARISON_RE = re.compile(
    r"\b(compare|versus|vs\.?|pros and cons|trade-?off|option)\b",
    re.I,
)
_EXPLICIT_STEPS_RE = re.compile(
    r"\b(step by step|steps?|sequence|first.+then)\b",
    re.I,
)

_EMPATHY_RESPONSE_RE = re.compile(
    r"\b(i hear you|i(?:'m| am) (?:really )?sorry|that sounds|"
    r"it makes sense (?:that|to)|understandably|you(?:'re| are) not alone|"
    r"this (?:is|sounds) (?:hard|painful|frustrating)|"
    r"thank you for (?:sharing|telling me))\b",
    re.I,
)
_ACTION_RESPONSE_RE = re.compile(
    r"\b(next step|start by|you can|try this|first|then|option|recommend|"
    r"check|test|measure|compare)\b",
    re.I,
)
_REASONING_RESPONSE_RE = re.compile(
    r"\b(because|therefore|however|trade-?off|hypothesis|evidence|verify|"
    r"if .+ then|first|second|finally|root cause|assumption)\b",
    re.I,
)
_CALIBRATION_RESPONSE_RE = re.compile(
    r"\b(i(?:'m| am) not certain|uncertain|may|might|likely|appears|based on|"
    r"confidence|estimate|could be wrong|i don't know|insufficient information)\b",
    re.I,
)
_VERIFY_RESPONSE_RE = re.compile(
    r"\b(verify|check (?:the|a) source|citation|evidence|confirm|cross-check|test)\b",
    re.I,
)
_CLARIFY_RESPONSE_RE = re.compile(
    r"\b(could you clarify|which .+ do you mean|what do you mean|"
    r"can you share|need more context)\b",
    re.I,
)
_UNEARNED_AGREEMENT_RE = re.compile(
    r"\b(you(?:'re| are) absolutely right|exactly right|i completely agree|"
    r"definitely,? you|yes,? obviously|without a doubt you)\b",
    re.I,
)
_OVERCLAIM_RE = re.compile(
    r"\b(always|never|guaranteed|definitely|obviously|certainly|everyone knows|"
    r"undeniably|100 percent|100%)\b",
    re.I,
)
_DISMISSIVE_RE = re.compile(
    r"\b(just get over it|calm down|no big deal|you're overreacting|stop worrying)\b",
    re.I,
)
_CRISIS_RESPONSE_RE = re.compile(
    r"\b((?:call|contact)\s+(?:local\s+)?emergency services|"
    r"go to (?:the )?(?:nearest )?emergency department|"
    r"(?:call|contact)\s+(?:a\s+)?crisis (?:line|service)|"
    r"(?:call|contact) emergency)\b",
    re.I,
)
_URGENT_HEALTH_RESPONSE_RE = re.compile(
    r"\b((?:call|contact)\s+(?:local\s+)?emergency services|"
    r"go to (?:the )?(?:nearest )?emergency department|"
    r"seek immediate (?:emergency|medical) care|"
    r"seek urgent medical (?:care|assessment)|"
    r"(?:call|contact) emergency)\b",
    re.I,
)
_UNSAFE_SAFETY_RESPONSE_RE = re.compile(
    r"\b(?:do not|don't|never|avoid|should not|shouldn't|would not|wouldn't|"
    r"must not|mustn't)\s+(?:(?:ever|immediately|directly)\s+)?"
    r"(?:call|contact|seek|go to|visit)\b"
    r".{0,50}\b(?:emergency|urgent|crisis|care|hospital)\b|"
    r"\b(?:no need to|you (?:do not|don't) need to)\s+"
    r"(?:call|contact|seek|go to|visit)\b.{0,50}"
    r"\b(?:emergency|urgent|crisis|care|hospital)\b|"
    r"\b(?:there is|there's)\s+no\s+(?:need|reason)\s+to\s+"
    r"(?:call|contact|seek|go to|visit)\b.{0,50}"
    r"\b(?:emergency|urgent|crisis|care|hospital)\b|"
    r"\b(?:it is|it's|that is|that's)\s+not\s+"
    r"(?:necessary|needed|recommended)\s+to\s+"
    r"(?:call|contact|seek|go to|visit)\b.{0,50}"
    r"\b(?:emergency|urgent|crisis|care|hospital)\b|"
    r"\b(?:i|you|we|they)\s+(?:do not|don't)\s+have\s+to\s+"
    r"(?:call|contact|seek|go to|visit)\b.{0,50}"
    r"\b(?:emergency|urgent|crisis|care|hospital)\b|"
    r"\b(?:rather than|instead of)\s+"
    r"(?:call|contact|seek|go to|visit)\b.{0,50}"
    r"\b(?:emergency|urgent|crisis|care|hospital)\b|"
    r"\b(?:wait until (?:tomorrow|later|morning)|wait and see|"
    r"wait for .{0,24} to pass|sleep it off|"
    r"see (?:if|whether) .{0,24}(?:passes|improves))\b",
    re.I,
)
_DISMISSIVE_EDUCATIONAL_RE = re.compile(
    r"\b(?:telling|saying)\s+(?:someone|a person|people)\b.{0,48}"
    r"\b(?:calm down|just get over it|no big deal|stop worrying)\b|"
    r"\b(?:do not|don't|never|avoid)\s+(?:tell|say)\b.{0,48}"
    r"\b(?:calm down|just get over it|no big deal|stop worrying)\b",
    re.I,
)
_SAFETY_CAUTION_RE = re.compile(
    r"\b(never mix|do not mix|don't mix|seek immediate|call emergency|"
    r"contact emergency|follow the safety|follow the manufacturer)\b",
    re.I,
)
_COMPARISON_RESPONSE_RE = re.compile(
    r"\b(whereas|compared with|versus|vs\.?|pros?|cons?|trade-?off|option [ab12])\b",
    re.I,
)
_STEP_RESPONSE_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+|\b(first|second|third|next|finally)\b",
    re.I,
)
_ASSUMPTION_RESPONSE_RE = re.compile(
    r"\b(assum(?:e|ed|ing|ption)|given that|conditional on|holding .+ constant|"
    r"under (?:a|the|this) model|if .+ then)\b",
    re.I,
)
_SCIENCE_OBSERVATION_RE = re.compile(
    r"\b(observation|observed|measurement|measured|data|evidence|result)\b",
    re.I,
)
_SCIENCE_TEST_RE = re.compile(
    r"\b(hypothesis|experiment|control group|independent variable|dependent variable|"
    r"replicat(?:e|ion)|falsif(?:y|iable)|test (?:whether|the)|inference)\b",
    re.I,
)
_FORECAST_STATEMENT_RE = re.compile(
    r"\b(forecast|predict(?:ion|ed)?|expected outcome|more likely|less likely|"
    r"most likely|conditional forecast)\b",
    re.I,
)
_FORECAST_BASIS_RE = re.compile(
    r"\b(range|scenario|credible interval|confidence interval|prediction interval|"
    r"base rate\s+(?:of|is|=)\s*\d|probability\s+(?:of|is|=|at)\s*\d|"
    r"odds\s+(?:of|are|=)\s*\d)\b",
    re.I,
)
_FORECAST_PERCENT_RE = re.compile(r"(?<![\w.])(?P<value>[+-]?\d+(?:\.\d+)?)\s*%")
_FORECAST_ASSUMPTION_BASIS_RE = re.compile(
    r"\b(?:independent|stationary|stable|constant|unchanged|exchangeable|"
    r"historical (?:data|process|rate|trend)|no (?:distribution|regime) shift|"
    r"same (?:distribution|process|success probability|rate)|base rate|seasonal pattern)\b",
    re.I,
)
_FORECAST_LIMIT_RE = re.compile(
    r"\b(?:not calibrated|uncalibrated|not (?:a )?guarantee|model[- ]conditional|"
    r"illustrative (?:estimate|scenario)|insufficient (?:data|evidence|information)|"
    r"cannot (?:estimate|forecast|predict)|can't (?:estimate|forecast|predict)|"
    r"need more (?:data|evidence)|would (?:make me )?abstain|subject to uncertainty)\b",
    re.I,
)
_ABSTENTION_RESPONSE_RE = re.compile(
    r"\b(insufficient (?:data|evidence|information)|cannot (?:estimate|forecast|predict)|"
    r"can't (?:estimate|forecast|predict)|need more (?:data|evidence)|abstain)\b",
    re.I,
)
_CALCULATION_VALUE_RE = re.compile(
    r"(?:[=≈~]\s*[+-]?\d|\b(?:result|answer)\s+(?:is|=)\s*[+-]?\d|"
    r"\b\d+(?:\.\d+)?\s*(?:n|j|w|v|a|ohms?|pa|k|mol|kg/m|g/cm|m/s|m\^3)\b)",
    re.I,
)
_CALCULATION_CHECK_RE = re.compile(
    r"\b(dimensional check|units? (?:match|cancel|are consistent)|substitut(?:e|ion) check|"
    r"inverse check|recompute|check(?:ed)? by)\b",
    re.I,
)
_CALCULATION_NUMBER_RE = re.compile(
    r"(?<![\w.^/])(?P<value>[+-]?(?:\d+\s*/\s*[+-]?\d+|"
    r"\d+(?:\.\d+)?(?:[eE][+-]?\d{1,4})?))"
    r"(?!\d|\.\d|/)(?P<percent>\s*%)?(?![\w/%])",
    re.I,
)
_CALCULATION_PERCENT_ROUNDING_TOLERANCE = Fraction(1, 200_000_000)
_CALCULATION_MAX_ABS_EXPONENT = 1_300
_CALCULATION_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:\b(?:answer|result|value|probability|chance|force|density|energy|voltage|"
    r"current|resistance|speed|distance|final\s+velocity|displacement|pressure|"
    r"volume|temperature|amount)\s+(?:is|=)|"
    r"\b(?:i\s+)?(?:predict|forecast|estimate)\s+(?:a\s+)?|"
    r"\bfinal(?:\s+answer)?\s*(?::|is|=)|"
    r"\b(?:i\s+)?(?:report|return|conclude)\s*(?::|is|=)?|"
    r"\b(?:therefore|thus|hence)\s+(?:final\s*:?)?)\s*$",
    re.I,
)
_CALCULATION_CLAUSE_ANSWER_RE = re.compile(
    r"\b(?:probability|chance)\s+(?:of|for)\s+[^.;?!\n]{0,96}?\s+"
    r"(?:is|=)\s*$",
    re.I,
)
_CALCULATION_POST_ASSERTION_RE = re.compile(
    r"^[^.;?!]{0,24}\b(?:is\s+correct|is\s+the\s+(?:answer|result)|"
    r"is\s+the\s+final\s+(?:answer|result))\b",
    re.I,
)
_CALCULATION_EQUATION_RESULT_RE = re.compile(r"(?:[=≈~]|\bequals?)\s*$", re.I)
_CALCULATION_REVISION_RE = re.compile(
    r"\b(?:correction|incorrect|instead|wrong|ignore\s+(?:it|that|this)|"
    r"actually\s+(?:ignore|use|the\s+answer)|not\s+the\s+final\s+(?:answer|result)|"
    r"(?:is|was)\s+false|reject\s+(?:it|that|this|the\s+value)|"
    r"do\s+not\s+use\s+(?:it|that|this)|allegedly|(?:answer|result|value)\s+fails?)\b",
    re.I,
)
_CALCULATION_NEGATED_ASSERTION_RE = re.compile(
    r"\b(?:wrong\s+to\s+(?:say|claim|report)|reject(?:ed|ing)?(?:\s+the\s+claim)?|"
    r"hypothetical|suppose|assuming\s+for\s+example|falsely|false\s+claim|"
    r"quoted|quoting|merely\s+(?:an?\s+)?example|not\s+(?:actually\s+)?the\s+answer)\b",
    re.I,
)
_CALCULATION_QUOTED_SPAN_RE = re.compile(
    r'''(?:```[\s\S]{0,8000}?```|`[^`\n]{0,4000}`|"[^"\n]{0,4000}"|'''
    r'''\u201c[^\u201d\n]{0,4000}\u201d|\u2018[^\u2019\n]{0,4000}\u2019|'''
    r'''\u00ab[^\u00bb\n]{0,4000}\u00bb|\u300c[^\u300d\n]{0,4000}\u300d|'''
    r'''\u300e[^\u300f\n]{0,4000}\u300f)''',
    re.I,
)
_CALCULATION_UNIT_ALIASES = {
    "n": r"(?:n|newtons?)\b",
    "j": r"(?:j|joules?)\b",
    "w": r"(?:w|watts?)\b",
    "v": r"(?:v|volts?)\b",
    "a": r"(?:a|amps?|amperes?)\b",
    "ω": r"(?:ω|ohms?)\b",
    "ohm": r"(?:ω|ohms?)\b",
    "pa": r"(?:pa|pascals?)\b",
    "k": r"(?:k|kelvins?)\b",
    "mol": r"(?:mol|moles?)\b",
    "m^3": r"(?:m\s*\^\s*3|m³|cubic\s+met(?:er|re)s?)\b",
    "%": r"%",
}
_CALCULATION_UNIT_EXTENSION_RE = re.compile(
    r"(?:\s*[/^*\u00b7\u22c5\u2022×-]\s*[A-Za-z0-9²³]|\s*[²³]|"
    r"\s+(?:per|seconds?|secs?|s|minutes?|mins?|hours?|h|meters?|metres?|m|"
    r"centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|kilometers?|"
    r"kilometres?|km|newtons?|n|joules?|j|watts?|w|volts?|v|amps?|amperes?|a|"
    r"ohms?|kilograms?|kg|grams?|g|liters?|litres?|l|feet|foot|ft|inches?|inch|"
    r"yards?|yd|miles?|mi)\b)",
    re.I,
)
_SCIENCE_CALCULATION_REQUEST_RE = re.compile(
    r"\b(?:what\s+is|find|calculate|determine|compute|solve\s+for)\s+"
    r"(?:its|the|an?)?\s*(?:final\s+velocity|displacement|pressure|volume|"
    r"temperature|amount(?:\s+of\s+substance)?)\b",
    re.I,
)
_SCIENCE_BOUNDARY_RESPONSE_RE = re.compile(
    r"\b(?:(?:cannot|can't|unable\s+to)\s+(?:safely\s+)?"
    r"(?:calculate|determine|verify|solve)|"
    r"(?:need|requires?)\s+(?:an?\s+)?explicit\s+"
    r"(?:(?:constant[- ]acceleration|ideal[- ]gas)\s+)?(?:model\s+)?assumption|"
    r"outside\s+(?:the\s+)?(?:supported|verified)\s+(?:model|scope)|"
    r"(?:consult|ask)\s+(?:a|an)\s+(?:qualified|licensed|domain)\s+"
    r"(?:professional|expert|clinician|engineer))\b",
    re.I,
)
_SCIENCE_INPUT_RECAP_RE = re.compile(
    r"\b(?:given|input|provided|reported|stated|supplied)\b",
    re.I,
)
_SCIENCE_ADVERSATIVE_RE = re.compile(r"\b(?:but|however|nevertheless|yet)\b", re.I)
_SCIENCE_RECAP_UNIT_RE = re.compile(
    r"\s*(?P<unit>(?:km\s*/\s*h(?:\s*(?:\^\s*2|\u00b2))?|"
    r"cm\s*/\s*s(?:\s*(?:\^\s*2|\u00b2))?|"
    r"m\s*/\s*s(?:\s*(?:\^\s*2|\u00b2))?|ft\s*/\s*s(?:\s*(?:\^\s*2|\u00b2))?|"
    r"mph|mpa|kpa|pa|pascals?|bars?|atm|atmospheres?|"
    r"m\s*(?:\^\s*3|\u00b3)|cm\s*(?:\^\s*3|\u00b3)|ml|lit(?:er|re)s?|l|"
    r"degrees?\s+celsius|celsius|\u00b0\s*c|kelvins?|k|"
    r"kmol|mmol|moles?|mol|seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|hr|h|"
    r"meters?|metres?|m))(?![A-Za-z0-9])",
    re.I,
)
_SCIENCE_POST_TARGET_ASSERTION_RE = re.compile(
    r"^[^.;?!\n]{0,40}\b(?:is|was|equals?)\s+(?:the\s+)?"
    r"(?:answer|result|final\s+velocity|displacement|pressure|volume|"
    r"temperature|amount(?:\s+of\s+substance)?)\b",
    re.I,
)
_SCIENCE_SPELLED_TARGET_ASSERTION_RE = re.compile(
    r"\b(?:answer|result|final\s+velocity|displacement|pressure|volume|"
    r"temperature|amount(?:\s+of\s+substance)?)\s+(?:is|was|equals?)\s+"
    r"(?:negative\s+|minus\s+)?(?:zero|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion)(?:[- ](?:one|two|three|four|five|six|"
    r"seven|eight|nine|ten|hundred|thousand|million|point))*\b",
    re.I,
)
_CAUSAL_MECHANISM_RE = re.compile(
    r"\b(causal|cause|effect|mechanism|mediator|because|leads? to|results? in)\b",
    re.I,
)
_CAUSAL_LIMIT_RE = re.compile(
    r"\b(alternative explanation|confound(?:er|ing)?|counterfactual|"
    r"correlation (?:is not|isn't|does not establish) causation|observational "
    r"(?:data|evidence)|cannot establish causality|reverse causality|selection bias)\b",
    re.I,
)

_CONTENT_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "could",
    "from",
    "have",
    "help",
    "into",
    "make",
    "more",
    "please",
    "should",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
    "anxious",
    "difficult",
    "feel",
    "feeling",
    "handle",
    "next",
    "overwhelmed",
    "sounds",
    "step",
    "together",
    "urgent",
}
_PROMPT_CONTROL_STOPWORDS = {
    "answer",
    "answers",
    "brief",
    "briefly",
    "constraint",
    "constraints",
    "exactly",
    "explain",
    "explanation",
    "format",
    "give",
    "heading",
    "headings",
    "respond",
    "response",
    "return",
    "sentence",
    "sentences",
    "step",
    "steps",
    "using",
    "write",
}
_ARITHMETIC_EQUATION_RE = re.compile(
    r"(?<!\w)[+-]?\d+(?:\.\d+)?\s*(?:[+\-*/=×÷])\s*"
    r"[+-]?\d+(?:\.\d+)?(?!\w)"
)
_FINAL_NUMERIC_ANSWER_RE = re.compile(
    r"\b(?:exact|final)\s+(?:answer|result)\s*(?:is|:)\s*"
    r"[+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?",
    re.I,
)
_ARITHMETIC_RESPONSE_LANGUAGE_RE = re.compile(
    r"\b(?:add|added|amounts?|arithmetic|calculate|combine|difference|divide|"
    r"multipl(?:y|ied|ier)|probability|product|quotient|subtract|sum|total)\b",
    re.I,
)
_ARITHMETIC_TASK_RE = re.compile(
    r"\b(?:add|arithmetic|average|calculate|compute|difference|divide|equation|"
    r"how many|minus|multiply|percentage|plus|probability|product|quotient|"
    r"ratio|solve|subtract|sum|times|total)\b|"
    r"(?<!\w)[+-]?\d+(?:\.\d+)?\s*(?:[+\-*/=×÷])\s*"
    r"[+-]?\d+(?:\.\d+)?(?!\w)",
    re.I,
)

_PROMPT_UNDERSTANDING_MODULE: Any = None
_REASONING_MODULE: Any = None
_GROUNDING_MODULE: Any = None
_PLANNER_INTENTS = {
    "conversation",
    "creative_generation",
    "decision_support",
    "editing",
    "emotional_support",
    "explanation",
    "factual_lookup",
    "problem_solving",
}
_OBJECTIVE_INTENT_MAP = {
    "answer": "explanation",
    "assess": "decision_support",
    "brainstorm": "creative_generation",
    "build": "problem_solving",
    "calculate": "problem_solving",
    "choose": "decision_support",
    "compare": "decision_support",
    "comfort": "emotional_support",
    "create": "creative_generation",
    "debug": "problem_solving",
    "decide": "decision_support",
    "edit": "editing",
    "explain": "explanation",
    "fix": "problem_solving",
    "generate": "creative_generation",
    "implement": "problem_solving",
    "invent": "creative_generation",
    "lookup": "factual_lookup",
    "plan": "problem_solving",
    "polish": "editing",
    "recommend": "decision_support",
    "research": "factual_lookup",
    "rewrite": "editing",
    "solve": "problem_solving",
    "summarize": "explanation",
    "support": "emotional_support",
    "teach": "explanation",
    "translate": "editing",
    "verify": "factual_lookup",
    "predict": "decision_support",
    "investigate": "problem_solving",
    "write": "creative_generation",
}
def _load_prompt_understanding_module() -> Any:
    """Load the mirrored sibling module even under file-based test imports."""

    global _PROMPT_UNDERSTANDING_MODULE
    if _PROMPT_UNDERSTANDING_MODULE is not None:
        return _PROMPT_UNDERSTANDING_MODULE
    try:
        import prompt_understanding as module
    except ImportError:
        module_path = Path(__file__).with_name("prompt_understanding.py")
        module_name = f"_supermix_{module_path.parent.name}_prompt_understanding"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load prompt understanding API from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, module)
        spec.loader.exec_module(module)
    _PROMPT_UNDERSTANDING_MODULE = module
    return module


def _load_reasoning_module() -> Any:
    """Load the bounded reasoning sibling for supported answer checks."""

    global _REASONING_MODULE
    if _REASONING_MODULE is not None:
        return _REASONING_MODULE
    try:
        import reasoning_engine as module
    except ImportError:
        module_path = Path(__file__).with_name("reasoning_engine.py")
        module_name = f"_supermix_{module_path.parent.name}_planner_reasoning"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, module)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
    _REASONING_MODULE = module
    return module


def _load_grounding_module() -> Any:
    """Load the sibling exact-arithmetic verifier without widening its grammar."""

    global _GROUNDING_MODULE
    if _GROUNDING_MODULE is not None:
        return _GROUNDING_MODULE
    module_path = Path(__file__).with_name("grounding_runtime.py")
    module_name = f"_supermix_{module_path.parent.name}_planner_grounding"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    _GROUNDING_MODULE = module
    return module


def _supported_calculation_result(user_text: str) -> Optional[Mapping[str, Any]]:
    """Resolve one answer through reasoning, then the exact arithmetic boundary."""

    module = _load_reasoning_module()
    solver = getattr(module, "solve_problem", None) if module is not None else None
    if callable(solver):
        try:
            result = solver(user_text)
        except Exception:
            result = None
        model_conditional_estimate = bool(
            isinstance(result, Mapping)
            and result.get("problem_class") == "prediction"
            and result.get("method") == "empirical_bernoulli_plugin"
            and result.get("reason") == "verified_non_overriding_estimate"
        )
        if isinstance(result, Mapping) and (
            bool(result.get("override_allowed")) or model_conditional_estimate
        ):
            verification = result.get("verification")
            if isinstance(verification, Mapping) and bool(verification.get("passed")):
                return result

    grounding = _load_grounding_module()
    arithmetic_solver = (
        getattr(grounding, "solve_exact_arithmetic", None)
        if grounding is not None
        else None
    )
    if not callable(arithmetic_solver):
        return None
    try:
        arithmetic = arithmetic_solver(user_text)
    except Exception:
        return None
    if not isinstance(arithmetic, Mapping) or arithmetic.get("solved") is not True:
        return None
    return {
        "override_allowed": True,
        "answer": {
            "exact": str(arithmetic.get("exact") or ""),
            "display": str(arithmetic.get("display") or ""),
            "approximation": "",
            "unit": "",
        },
        "verification": {"passed": True},
    }


def _looks_like_science_calculation(text: str, scientific_request: bool) -> bool:
    """Recognize bounded-physics shapes even when an assumption is missing."""

    value = str(text or "")
    if not re.search(r"\d", value) or _SCIENCE_CALCULATION_REQUEST_RE.search(value) is None:
        return False
    kinematics_shape = all(
        re.search(pattern, value, re.I) is not None
        for pattern in (r"\binitial\s+velocity\b", r"\bacceleration\b", r"\btime\b")
    )
    gas_amount = re.search(
        r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d{1,3})?\s*"
        r"(?:kmol|mmol|mol(?:es)?)\b",
        value,
        re.I,
    ) is not None
    gas_labels = sum(
        re.search(rf"\b{label}\b", value, re.I) is not None
        for label in ("pressure", "volume", "temperature")
    )
    return bool(scientific_request or kinematics_shape or (gas_amount and gas_labels >= 2))


def _science_quantity_token(text: str, match: re.Match[str]) -> Optional[Tuple[Fraction, str]]:
    raw_value = match.group("value").replace(" ", "")
    exponent_match = re.search(r"[eE](?P<exponent>[+-]?\d+)$", raw_value)
    if (
        exponent_match is not None
        and abs(int(exponent_match.group("exponent")))
        > _CALCULATION_MAX_ABS_EXPONENT
    ):
        return None
    unit_match = _SCIENCE_RECAP_UNIT_RE.match(text[match.end("value") :])
    if unit_match is None:
        return None
    try:
        value = Fraction(raw_value)
    except (ValueError, ZeroDivisionError):
        return None
    unit = re.sub(r"\s+", "", unit_match.group("unit")).casefold()
    unit = unit.replace("\u00b2", "^2").replace("\u00b3", "^3")
    return value, unit


def _unsupported_science_answer_asserted(response_text: str, user_text: str) -> bool:
    """Reject derived numbers while allowing prompt-bound input recaps."""

    prompt = str(user_text or "")
    prompt_tokens = {
        token
        for match in _CALCULATION_NUMBER_RE.finditer(prompt)
        if (token := _science_quantity_token(prompt, match)) is not None
    }
    text = str(response_text or "")
    if _SCIENCE_SPELLED_TARGET_ASSERTION_RE.search(text) is not None:
        return True
    for match in _CALCULATION_NUMBER_RE.finditer(text):
        clause_start = max(
            text.rfind(delimiter, 0, match.start())
            for delimiter in (".", ";", "?", "!", "\n")
        ) + 1
        following_delimiters = [
            position
            for delimiter in (".", ";", "?", "!", "\n")
            if (position := text.find(delimiter, match.end())) >= 0
        ]
        clause_end = min(following_delimiters) if following_delimiters else len(text)
        clause = text[clause_start:clause_end]
        prefix = text[clause_start:match.start()]
        suffix = text[match.end() : match.end() + 80]
        if _CALCULATION_NEGATED_ASSERTION_RE.search(prefix) is not None:
            continue
        token = _science_quantity_token(text, match)
        if token is None:
            if (
                _CALCULATION_EXPLICIT_ANSWER_RE.search(prefix) is not None
                or _CALCULATION_POST_ASSERTION_RE.search(
                    suffix
                )
                is not None
                or _SCIENCE_POST_TARGET_ASSERTION_RE.search(suffix) is not None
            ):
                return True
            continue
        if (
            token in prompt_tokens
            and _SCIENCE_INPUT_RECAP_RE.search(clause) is not None
            and _SCIENCE_ADVERSATIVE_RE.search(clause) is None
            and _CALCULATION_EXPLICIT_ANSWER_RE.search(prefix) is None
            and _SCIENCE_POST_TARGET_ASSERTION_RE.search(suffix) is None
        ):
            continue
        return True
    return False


def _calculation_unit_matches(text: str, end: int, expected_unit: str) -> bool:
    """Check the unit immediately following one candidate answer value."""

    unit = str(expected_unit or "").strip()
    if not unit:
        return True
    canonical = unit.lower().replace("ohms", "ohm").replace("ohm", "ohm")
    pattern = _CALCULATION_UNIT_ALIASES.get(canonical)
    if pattern is None:
        # Preserve exact compound-unit spelling while allowing ordinary spacing.
        escaped = re.escape(unit)
        escaped = escaped.replace(r"\ ", r"\s*")
        pattern = escaped + r"(?![A-Za-z0-9])"
    unit_match = re.match(r"\s*(?:" + pattern + r")", text[end:], re.I)
    if unit_match is None:
        return False
    remaining = text[end + unit_match.end() :]
    return _CALCULATION_UNIT_EXTENSION_RE.match(remaining) is None


def _candidate_matches_verified_calculation(
    response_text: str,
    user_text: str,
) -> Optional[bool]:
    """Check a numeric answer when the bounded reasoning engine supports it.

    ``None`` means the deterministic engine cannot adjudicate this prompt, so
    the planner cannot grant a capability named ``verified_calculation``.
    ``False`` is emitted for a solver-backed answer that does not match a
    response presenting itself as a checked calculation.
    """

    result = _supported_calculation_result(user_text)
    model_conditional_estimate = bool(
        isinstance(result, Mapping)
        and result.get("problem_class") == "prediction"
        and result.get("method") == "empirical_bernoulli_plugin"
        and result.get("reason") == "verified_non_overriding_estimate"
    )
    if not isinstance(result, Mapping) or not (
        bool(result.get("override_allowed")) or model_conditional_estimate
    ):
        return None
    verification = result.get("verification")
    if not isinstance(verification, Mapping) or not bool(verification.get("passed")):
        return None
    answer = result.get("answer")
    if not isinstance(answer, Mapping):
        return None

    expected_values = set()
    exact_value: Optional[Fraction] = None
    for field in ("exact", "display", "approximation"):
        raw_value = str(answer.get(field) or "").strip()
        if not raw_value:
            continue
        try:
            parsed = Fraction(raw_value.replace(" ", ""))
        except (ValueError, ZeroDivisionError):
            continue
        expected_values.add(parsed)
        if field == "exact":
            exact_value = parsed
    if not expected_values:
        return None

    text = str(response_text or "")
    expected_unit = str(answer.get("unit") or "").strip()
    quoted_spans = [
        (match.start(), match.end())
        for match in _CALCULATION_QUOTED_SPAN_RE.finditer(text)
    ]
    explicit_answers = []
    equation_answers = []
    answer_assertions = []
    for match in _CALCULATION_NUMBER_RE.finditer(text):
        if any(start <= match.start() < end for start, end in quoted_spans):
            continue
        candidate_text = match.group("value").replace(" ", "")
        exponent_match = re.search(
            r"[eE](?P<exponent>[+-]?\d+)$",
            candidate_text,
        )
        if (
            exponent_match is not None
            and abs(int(exponent_match.group("exponent")))
            > _CALCULATION_MAX_ABS_EXPONENT
        ):
            continue
        try:
            candidate = Fraction(candidate_text)
        except (ValueError, ZeroDivisionError):
            continue
        if match.group("percent") and expected_unit != "%":
            candidate /= 100
        rounded_percent_matches = bool(
            match.group("percent")
            and expected_unit != "%"
            and exact_value is not None
            and abs(candidate - exact_value)
            <= _CALCULATION_PERCENT_ROUNDING_TOLERANCE
        )
        unit_matches = _calculation_unit_matches(text, match.end("value"), expected_unit)
        value_matches = candidate in expected_values or rounded_percent_matches
        answer_matches = bool(value_matches and (not expected_unit or unit_matches))
        prefix = text[max(0, match.start() - 32) : match.start()]
        clause_start = max(
            text.rfind(delimiter, 0, match.start())
            for delimiter in (".", ";", "?", "!", "\n")
        ) + 1
        clause_prefix = text[max(clause_start, match.start() - 160) : match.start()]
        if _CALCULATION_NEGATED_ASSERTION_RE.search(
            text[clause_start : match.start()]
        ) is not None:
            continue
        post_assertion = _CALCULATION_POST_ASSERTION_RE.search(
            text[match.end() : match.end() + 40]
        )
        is_explicit = bool(
            _CALCULATION_EXPLICIT_ANSWER_RE.search(prefix) is not None
            or _CALCULATION_CLAUSE_ANSWER_RE.search(clause_prefix) is not None
            or post_assertion is not None
        )
        is_equation = bool(
            not is_explicit
            and _CALCULATION_EQUATION_RESULT_RE.search(prefix) is not None
        )
        assertion = (answer_matches, match.end(), is_explicit, is_equation)
        if is_explicit:
            explicit_answers.append(assertion)
        elif is_equation:
            equation_answers.append(assertion)
        answer_assertions.append(assertion)
    if explicit_answers:
        last_explicit_matches, last_end, _, _ = explicit_answers[-1]
        if _CALCULATION_REVISION_RE.search(text[last_end:]) is not None:
            return False
        return bool(
            last_explicit_matches
            and all(
                assertion[0]
                for assertion in answer_assertions
                if assertion[1] >= last_end
            )
        )
    if _CALCULATION_REVISION_RE.search(text) is not None:
        return False
    if equation_answers:
        last_equation_matches, last_equation_end, _, _ = equation_answers[-1]
        return bool(
            last_equation_matches
            and all(
                assertion[0]
                for assertion in answer_assertions
                if assertion[1] >= last_equation_end
            )
        )
    return False


def _turn_messages(raw_turns: Any) -> tuple[list[Any], list[str], list[str]]:
    turns = list(raw_turns) if isinstance(raw_turns, (list, tuple)) else []
    users: list[str] = []
    assistants: list[str] = []
    for turn in turns:
        user = ""
        assistant = ""
        if isinstance(turn, Mapping):
            user = str(
                turn.get("user")
                or turn.get("user_text")
                or turn.get("prompt")
                or ""
            ).strip()
            assistant = str(
                turn.get("assistant")
                or turn.get("assistant_text")
                or turn.get("response")
                or ""
            ).strip()
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user = str(turn[0] or "").strip()
            assistant = str(turn[1] or "").strip()
        if user:
            users.append(user)
        if assistant:
            assistants.append(assistant)
    return turns, users, assistants


def _resolve_prompt_profile(
    text: str,
    prompt_profile: Optional[Mapping[str, Any]],
    *,
    recent_turns: Sequence[Any],
    recent_user_messages: Sequence[str],
    recent_assistant_messages: Sequence[str],
) -> Dict[str, Any]:
    if isinstance(prompt_profile, Mapping):
        return dict(prompt_profile)
    module = _load_prompt_understanding_module()
    profile = module.analyze_prompt(
        text,
        recent_turns=recent_turns,
        recent_user_messages=recent_user_messages,
        recent_assistant_messages=recent_assistant_messages,
    )
    return dict(profile) if isinstance(profile, Mapping) else {}


def _profile_intent(profile: Mapping[str, Any]) -> tuple[str, float]:
    best_intent = ""
    best_confidence = -1.0
    objectives = profile.get("objectives")
    if not isinstance(objectives, (list, tuple)):
        return best_intent, best_confidence
    for objective in objectives:
        if not isinstance(objective, Mapping):
            continue
        if str(objective.get("mode") or "required").lower() == "forbidden":
            continue
        act = str(objective.get("act") or "").strip().lower()
        if act == "conversation":
            # This is the prompt parser's fallback objective, not an explicit
            # signal strong enough to displace the planner's intent heuristics.
            continue
        mapped = act if act in _PLANNER_INTENTS else _OBJECTIVE_INTENT_MAP.get(act, "")
        if not mapped:
            continue
        try:
            confidence = float(objective.get("confidence", 0.0))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        confidence = _clamp(confidence)
        if confidence > best_confidence:
            best_intent = mapped
            best_confidence = confidence
    return best_intent, best_confidence


def _profile_capabilities(
    profile: Mapping[str, Any],
    key: str,
) -> list[str]:
    contract = profile.get("response_contract")
    raw = contract.get(key) if isinstance(contract, Mapping) else ()
    if not isinstance(raw, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in raw
            if str(item).strip()
        )
    )


def _profile_clarification(
    profile: Mapping[str, Any],
    text: str,
) -> Dict[str, Any]:
    ambiguity = profile.get("ambiguity")
    ambiguity = dict(ambiguity) if isinstance(ambiguity, Mapping) else {}
    conflicts = profile.get("conflicts")
    conflicts = conflicts if isinstance(conflicts, (list, tuple)) else ()
    references = profile.get("references")
    references = references if isinstance(references, (list, tuple)) else ()
    objectives = profile.get("objectives")
    objectives = objectives if isinstance(objectives, (list, tuple)) else ()

    target_refs = {
        str(item.get("target_ref") or "").strip()
        for item in objectives
        if isinstance(item, Mapping)
        and str(item.get("mode") or "required").lower() != "forbidden"
        and str(item.get("target_ref") or "").strip()
    }
    unresolved_candidates = [
        item
        for item in references
        if isinstance(item, Mapping)
        and str(item.get("status") or "").lower() in {"unresolved", "ambiguous"}
        and (
            not target_refs
            or str(item.get("id") or "").strip() in target_refs
        )
    ]
    required_acts = {
        str(item.get("act") or "").lower()
        for item in objectives
        if isinstance(item, Mapping)
        and str(item.get("mode") or "required").lower() != "forbidden"
    }
    reference_action = bool(
        required_acts & {"edit", "translate", "summarize"}
        or re.search(
            r"\b(?:make|do|fix|rewrite|edit|improve|change|expand)\s+"
            r"(?:it|this|that)\b|\bwhat\s+about\s+(?:it|this|that)\b",
            str(text or ""),
            re.I,
        )
    )
    unresolved = unresolved_candidates if reference_action else []
    blocking_conflicts = [
        item
        for item in conflicts
        if isinstance(item, Mapping)
        and (
            bool(item.get("blocking"))
            or str(item.get("severity") or "").lower() == "hard"
        )
    ]
    clarification_required = bool(
        unresolved
        or blocking_conflicts
        or (
            bool(ambiguity.get("clarification_required"))
            and not references
            and not conflicts
        )
    )
    if blocking_conflicts:
        reason = "hard_constraint_conflict"
        question = "Which of the conflicting requirements should take priority?"
    elif unresolved:
        reason = "unresolved_required_reference"
        question = (
            "What should I apply this request to—the previous response, "
            "a file, or another item?"
        )
    else:
        reason = "prompt_ambiguity"
        question = "Which specific result do you want me to produce?"
    return {
        "required": clarification_required,
        "reason": reason if clarification_required else "",
        "question": question if clarification_required else "",
        "unresolved_reference_count": len(unresolved),
        "hard_conflict_count": len(blocking_conflicts),
    }


def _profile_constraint_audit(
    response_text: str,
    prompt_text: str,
    profile: Any,
) -> Dict[str, Any]:
    if not isinstance(profile, Mapping):
        return {
            "accepted": True,
            "checked_constraint_ids": [],
            "passed_constraint_ids": [],
            "violations": [],
            "unchecked_constraint_ids": [],
            "coverage": 1.0,
        }
    module = _load_prompt_understanding_module()
    audit = module.evaluate_response_constraints(
        str(response_text or ""),
        str(prompt_text or ""),
        profile,
    )
    return dict(audit) if isinstance(audit, Mapping) else {
        "accepted": True,
        "checked_constraint_ids": [],
        "passed_constraint_ids": [],
        "violations": [],
        "unchecked_constraint_ids": [],
        "coverage": 1.0,
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _match_score(pattern: re.Pattern[str], text: str, scale: float) -> float:
    return _clamp(len(pattern.findall(text)) * scale)


def _assumption_response_score(text: str) -> float:
    """Require an assumption with content, not a magic phrase such as 'a model'."""

    ignored = {
        "assume", "assumed", "assuming", "assumption", "conditional", "given",
        "holding", "under", "model", "this", "that", "then", "with", "the",
    }
    for match in _ASSUMPTION_RESPONSE_RE.finditer(str(text or "")):
        end = min(len(text), match.end() + 96)
        span = re.split(r"[.,;]", text[match.start() : end], maxsplit=1)[0]
        meaningful = [
            token
            for token in re.findall(r"[a-z][a-z'-]{2,}", span.lower())
            if token not in ignored
        ]
        if len(set(meaningful)) >= 2:
            return 0.30
    return 0.0


def _forecast_basis_score(text: str) -> float:
    """Accept bounded probabilities or an explicit range/scenario basis."""

    percentages = [
        float(match.group("value"))
        for match in _FORECAST_PERCENT_RE.finditer(str(text or ""))
    ]
    if percentages:
        if any(not 0.0 <= value <= 100.0 for value in percentages):
            return 0.0
        return _clamp(len(percentages) * 0.24)
    return _match_score(_FORECAST_BASIS_RE, str(text or ""), 0.24)


def _multi_part_coverage_score(text: str, expected: Any) -> float:
    """Check for observable per-part structure without claiming semantic truth."""

    try:
        target = max(0, min(8, int(expected or 0)))
    except (TypeError, ValueError, OverflowError):
        target = 0
    if target < 2:
        return 0.0
    raw = str(text or "")
    list_items = len(
        re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", raw)
    )
    equation_answers = len(
        re.findall(r"(?:^|[;,.\n])\s*[^;,.\n]{0,48}[=≈]\s*[+-]?\d", raw)
    )
    labelled = len(
        set(
            re.findall(
                r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\b",
                raw,
                re.I,
            )
        )
    )
    return 0.30 if max(list_items, equation_answers, labelled) >= target else 0.0


def _safety_support_score(
    pattern: re.Pattern[str],
    text: str,
    scale: float,
) -> float:
    if _UNSAFE_SAFETY_RESPONSE_RE.search(str(text or "")):
        return 0.0
    return _match_score(pattern, text, scale)


def _unearned_agreement_score(text: str) -> float:
    visible = _QUOTED_SPAN_RE.sub(" ", str(text or ""))
    if re.search(
        r"\b(?:do not|don't|never|avoid)\s+(?:say|tell|agree)\b",
        visible,
        re.I,
    ):
        return 0.0
    return _match_score(_UNEARNED_AGREEMENT_RE, visible, 0.75)


def _dismissive_score(text: str) -> float:
    raw = str(text or "")
    visible = _QUOTED_SPAN_RE.sub(" ", raw)
    if _DISMISSIVE_EDUCATIONAL_RE.search(raw):
        return 0.0
    return _match_score(_DISMISSIVE_RE, visible, 0.80)


def _turn_affect(text: str) -> Dict[str, float]:
    negative = _match_score(_EMOTIONAL_RE, text, 0.20)
    positive = _match_score(_POSITIVE_RE, text, 0.24)
    arousal = _clamp(_match_score(_HIGH_AROUSAL_RE, text, 0.42) + 0.18 * negative)
    distress = _clamp(
        0.70 * negative
        + 0.35 * arousal
        + _match_score(_SUPPORT_RE, text, 0.38)
    )
    return {
        "negative": negative,
        "positive": positive,
        "arousal": arousal,
        "distress": distress,
        "valence": _clamp(0.5 + 0.5 * positive - 0.5 * negative),
    }


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", str(text or "").lower())
        if len(token) >= 4 and token not in _CONTENT_STOPWORDS
    }


def _lexical_relevance(query_text: str, response_text: str) -> float:
    query_terms = _content_terms(query_text)
    if not query_terms:
        return 1.0
    response_terms = _content_terms(response_text)
    return _clamp(
        len(query_terms & response_terms) / max(1.0, min(6.0, len(query_terms)))
    )


def _topic_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9'-]*", str(text or "").lower())
        if len(token) >= 4
        and token not in _CONTENT_STOPWORDS
        and token not in _PROMPT_CONTROL_STOPWORDS
    }


def _looks_like_arithmetic_template(text: str) -> bool:
    raw = str(text or "")
    equation_count = len(_ARITHMETIC_EQUATION_RE.findall(raw))
    return bool(
        _FINAL_NUMERIC_ANSWER_RE.search(raw)
        or equation_count >= 2
        or (
            equation_count >= 1
            and _ARITHMETIC_RESPONSE_LANGUAGE_RE.search(raw)
        )
    )


def _incompatible_arithmetic_template(
    response_text: str,
    user_text: str,
    interaction_plan: Mapping[str, Any],
    relevance_context: str,
) -> bool:
    if not _looks_like_arithmetic_template(response_text):
        return False
    profile = interaction_plan.get("prompt_profile")
    profile = dict(profile) if isinstance(profile, Mapping) else {}
    prompt_context = profile.get("context")
    prompt_context = (
        dict(prompt_context) if isinstance(prompt_context, Mapping) else {}
    )
    followup = bool(prompt_context.get("followup"))
    if followup and not str(relevance_context or "").strip():
        return False
    query_context = " ".join(
        part.strip()
        for part in (
            str(relevance_context or "") if followup else "",
            str(user_text or ""),
        )
        if part and part.strip()
    )
    if _ARITHMETIC_TASK_RE.search(query_context):
        return False
    query_terms = _topic_terms(query_context)
    if len(query_terms) < 2:
        return False
    return not bool(query_terms & _topic_terms(response_text))


def _intent_scores(text: str) -> Dict[str, float]:
    emotional = max(
        _match_score(_EMOTIONAL_RE, text, 0.30),
        _match_score(_SUPPORT_RE, text, 0.50),
    )
    scores = {
        "emotional_support": emotional,
        "problem_solving": _match_score(_PROBLEM_RE, text, 0.28),
        "decision_support": _match_score(_DECISION_RE, text, 0.34),
        "explanation": _match_score(_EXPLAIN_RE, text, 0.30),
        "creative_generation": _match_score(_CREATIVE_RE, text, 0.28),
        "editing": _match_score(_EDIT_RE, text, 0.34),
        "factual_lookup": _match_score(_LOOKUP_RE, text, 0.28),
        "conversation": 0.08,
    }
    if "?" in text:
        scores["explanation"] += 0.06
    if re.search(
        r"^\s*(please\s+)?(?:help|solve|fix|make|build|find|give|show)\b",
        text,
        re.I,
    ):
        scores["problem_solving"] += 0.08
    return {key: _clamp(value) for key, value in scores.items()}


def _choose_strategy(
    primary_intent: str,
    distress: float,
    ambiguity: float,
    factuality: float,
) -> Dict[str, str]:
    if ambiguity >= 0.72:
        return {
            "response_strategy": "clarify_then_act",
            "reasoning_mode": "targeted_clarification",
        }
    if primary_intent == "emotional_support" or distress >= 0.30:
        return {
            "response_strategy": "validate_then_help",
            "reasoning_mode": "reflective_support",
        }
    if primary_intent == "problem_solving":
        return {
            "response_strategy": "decompose_test_then_recommend",
            "reasoning_mode": "deliberate_problem_solving",
        }
    if primary_intent == "decision_support":
        return {
            "response_strategy": "compare_tradeoffs_then_choose",
            "reasoning_mode": "comparative_reasoning",
        }
    if primary_intent == "creative_generation":
        return {
            "response_strategy": "diverge_then_refine",
            "reasoning_mode": "creative_exploration",
        }
    if primary_intent == "editing":
        return {
            "response_strategy": "preserve_intent_then_revise",
            "reasoning_mode": "constraint_tracking",
        }
    if primary_intent == "factual_lookup" or factuality >= 0.52:
        return {
            "response_strategy": "answer_with_evidence_and_uncertainty",
            "reasoning_mode": "verification_first",
        }
    if primary_intent == "explanation":
        return {
            "response_strategy": "explain_then_check_understanding",
            "reasoning_mode": "causal_explanation",
        }
    return {
        "response_strategy": "direct_then_offer_depth",
        "reasoning_mode": "direct",
    }


def plan_interaction(
    query_text: str,
    recent_assistant_messages: Sequence[str] = (),
    context: Optional[Mapping[str, Any]] = None,
    prompt_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable plan from bounded, observable request cues."""

    context = dict(context or {})
    text = str(query_text or "").strip()
    recent_turns, turn_users, turn_assistants = _turn_messages(
        context.get("recent_turns", ())
    )
    raw_recent_users = context.get("recent_user_messages", ()) or ()
    if not isinstance(raw_recent_users, (list, tuple)):
        raw_recent_users = ()
    recent_users = list(
        dict.fromkeys(
            str(message).strip()
            for message in [*raw_recent_users, *turn_users]
            if str(message).strip()
        )
    )[-4:]
    combined_assistants = list(
        dict.fromkeys(
            str(message).strip()
            for message in [*recent_assistant_messages, *turn_assistants]
            if str(message).strip()
        )
    )[-4:]
    profile = _resolve_prompt_profile(
        text,
        prompt_profile,
        recent_turns=recent_turns,
        recent_user_messages=recent_users,
        recent_assistant_messages=combined_assistants,
    )
    reasoning_profile = (
        dict(profile.get("reasoning") or {})
        if isinstance(profile.get("reasoning"), Mapping)
        else {}
    )
    scientific_request = bool(reasoning_profile.get("scientific"))
    mathematical_request = bool(reasoning_profile.get("mathematical"))
    science_calculation_shape = _looks_like_science_calculation(
        text,
        scientific_request,
    )
    solver_backed_science = bool(
        science_calculation_shape
        and isinstance(_supported_calculation_result(text), Mapping)
    )
    unsupported_science_calculation = bool(
        science_calculation_shape and not solver_backed_science
    )
    quantitative_request = bool(
        solver_backed_science
        or (
            reasoning_profile.get("verification_required")
            and not unsupported_science_calculation
        )
    )
    investigation_request = bool(reasoning_profile.get("investigative"))
    prediction_request = bool(reasoning_profile.get("predictive"))
    causal_request = bool(reasoning_profile.get("causal"))
    conversation_request = bool(reasoning_profile.get("conversational"))
    multi_part_expected = 0
    if bool(reasoning_profile.get("multi_part")):
        try:
            multi_part_expected = max(
                2,
                min(8, int(reasoning_profile.get("question_count") or 0)),
            )
        except (TypeError, ValueError, OverflowError):
            multi_part_expected = 2
    scores = _intent_scores(text)
    profile_intent, profile_intent_confidence = _profile_intent(profile)
    primary_intent = profile_intent or max(scores, key=scores.get)
    if profile_intent:
        scores[profile_intent] = max(
            scores.get(profile_intent, 0.0),
            profile_intent_confidence,
        )
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    intent_confidence = _clamp(0.42 + 0.48 * ordered[0] + 0.35 * margin)
    if profile_intent:
        intent_confidence = max(intent_confidence, profile_intent_confidence)

    affect = _turn_affect(text)
    current_distress = affect["distress"]
    uncertainty = _clamp(
        (0.28 if "?" in text else 0.0)
        + _match_score(
            re.compile(
                r"\b(not sure|uncertain|confused|maybe|i think|could it be)\b",
                re.I,
            ),
            text,
            0.28,
        )
    )
    word_count = len(re.findall(r"\b\w+\b", text))
    has_anchor = bool(combined_assistants)
    heuristic_ambiguity = 0.0
    if word_count <= 4 and _DEICTIC_RE.search(text):
        heuristic_ambiguity += 0.58
    if _DEICTIC_RE.search(text) and not has_anchor:
        heuristic_ambiguity += 0.34
    heuristic_ambiguity = _clamp(heuristic_ambiguity)
    profile_ambiguity = profile.get("ambiguity")
    profile_ambiguity = (
        dict(profile_ambiguity)
        if isinstance(profile_ambiguity, Mapping)
        else {}
    )
    raw_profile_ambiguity = profile_ambiguity.get("score")
    try:
        ambiguity = _clamp(float(raw_profile_ambiguity))
    except (TypeError, ValueError, OverflowError):
        ambiguity = heuristic_ambiguity
    if bool(profile_ambiguity.get("clarification_required")):
        ambiguity = max(ambiguity, 0.82)
    elif str(profile_ambiguity.get("status") or "").lower() == "clear":
        ambiguity = min(ambiguity, 0.20)
    clarification = _profile_clarification(profile, text)
    if (
        not clarification["required"]
        and str(profile_ambiguity.get("status") or "").lower()
        == "clarification_required"
    ):
        ambiguity = heuristic_ambiguity
    history_distress = 0.0
    if recent_users:
        values = [_turn_affect(message)["distress"] for message in recent_users]
        weights = list(range(1, len(values) + 1))
        history_distress = sum(
            weight * value for weight, value in zip(weights, values)
        ) / float(sum(weights))
    anchored_followup = bool(
        has_anchor
        and word_count <= 12
        and (
            _DEICTIC_RE.search(text)
            or _AFFECT_CONTINUITY_RE.search(text)
            or _SUPPORT_RE.search(text)
        )
        and history_distress >= 0.30
    )
    distress = _clamp(
        max(current_distress, (0.65 if anchored_followup else 0.25) * history_distress)
    )
    affect_delta = current_distress - history_distress
    affect_trend = "insufficient_context"
    if recent_users:
        if affect_delta >= 0.18:
            affect_trend = "possibly_rising"
        elif affect_delta <= -0.18:
            affect_trend = "possibly_easing"
        else:
            affect_trend = "roughly_stable"

    factuality = _clamp(
        0.48 * scores["factual_lookup"]
        + (0.42 if _HIGH_STAKES_RE.search(text) else 0.0)
        + (0.28 if _RECENCY_RE.search(text) else 0.0)
        + (0.14 if re.search(r"\b\d+(?:\.\d+)?\b", text) else 0.0)
        + (0.20 if scientific_request else 0.0)
        + (0.16 if prediction_request else 0.0)
    )
    agreement_request = bool(_AGREEMENT_REQUEST_RE.search(text))
    certainty_framing = bool(_CERTAINTY_FRAMING_RE.search(text))
    sycophancy = _clamp(
        (0.82 if agreement_request else 0.0)
        + (0.18 if certainty_framing and "?" not in text else 0.0)
        + (0.10 if "right?" in text.lower() else 0.0)
    )
    unquoted_safety_text = _QUOTED_SPAN_RE.sub(" ", text)
    reported_safety_context = bool(_REPORTED_SAFETY_CONTEXT_RE.search(text))
    current_disclosure_override = bool(
        _CURRENT_DISCLOSURE_OVERRIDE_RE.search(unquoted_safety_text)
    )
    heuristic_crisis_signal = bool(
        _PERSONAL_CRISIS_RE.search(unquoted_safety_text)
        and (not reported_safety_context or current_disclosure_override)
    )
    heuristic_urgent_health_signal = bool(
        _PERSONAL_URGENT_HEALTH_RE.search(unquoted_safety_text)
        and (not reported_safety_context or current_disclosure_override)
    )
    profile_safety = profile.get("safety")
    profile_safety = (
        dict(profile_safety) if isinstance(profile_safety, Mapping) else {}
    )
    profile_crisis_signal = bool(
        profile_safety.get("personal_crisis_signal")
        and not _NONCRISIS_IDIOM_RE.search(text)
    )
    crisis_signal = bool(heuristic_crisis_signal or profile_crisis_signal)
    urgent_health_signal = bool(
        heuristic_urgent_health_signal
        or profile_safety.get("urgent_health_signal")
    )

    strategy = _choose_strategy(primary_intent, distress, ambiguity, factuality)
    if prediction_request:
        strategy = {
            "response_strategy": "assumptions_then_conditional_forecast",
            "reasoning_mode": "probabilistic_reasoning",
        }
    elif investigation_request:
        strategy = {
            "response_strategy": "hypothesis_evidence_test_then_conclude",
            "reasoning_mode": "scientific_reasoning",
        }
    elif quantitative_request:
        strategy = {
            "response_strategy": "derive_verify_then_answer",
            "reasoning_mode": "quantitative_reasoning",
        }
    elif causal_request:
        strategy = {
            "response_strategy": "mechanism_alternatives_then_test",
            "reasoning_mode": "causal_reasoning",
        }
    elif conversation_request:
        strategy = {
            "response_strategy": "preserve_context_then_answer",
            "reasoning_mode": "conversation_tracking",
        }
    if clarification["required"]:
        strategy = {
            "response_strategy": "clarify_then_act",
            "reasoning_mode": "targeted_clarification",
        }
    if crisis_signal:
        strategy = {
            "response_strategy": "crisis_support_then_immediate_help",
            "reasoning_mode": "safety_first",
        }
    elif urgent_health_signal:
        strategy = {
            "response_strategy": "urgent_health_escalation",
            "reasoning_mode": "safety_first",
        }

    complexity = _clamp(
        (0.34 if primary_intent in {"problem_solving", "decision_support"} else 0.0)
        + (0.18 if primary_intent == "explanation" else 0.0)
        + (0.20 if _MULTISTEP_RE.search(text) else 0.0)
        + (0.18 if _CODE_MATH_RE.search(text) else 0.0)
        + (0.16 if mathematical_request else 0.0)
        + (0.18 if scientific_request else 0.0)
        + (0.16 if prediction_request or causal_request else 0.0)
        + min(0.18, 0.06 * len(_CONSTRAINT_RE.findall(text)))
        + (0.12 if word_count >= 30 else 0.0)
    )
    epistemic_risk = _clamp(
        max(factuality, 0.82 if _HIGH_STAKES_RE.search(text) else 0.0)
    )
    value_of_compute = _clamp(
        0.58 * complexity + 0.32 * epistemic_risk + 0.10 * uncertainty
    )
    deliberation_reasons = []
    if complexity >= 0.48:
        deliberation_reasons.append("task_complexity")
    if epistemic_risk >= 0.34:
        deliberation_reasons.append("epistemic_risk")
    if uncertainty >= 0.48:
        deliberation_reasons.append("user_uncertainty")
    if crisis_signal or urgent_health_signal:
        suggested_floor = 1
        deliberation_reasons = ["safety_fast_path"]
    elif ambiguity >= 0.72:
        suggested_floor = 1
        deliberation_reasons.append("clarify_before_deep_compute")
    elif complexity >= 0.30 or epistemic_risk >= 0.34 or value_of_compute >= 0.30:
        suggested_floor = 3
    else:
        suggested_floor = 1
    if not deliberation_reasons:
        deliberation_reasons.append("low_risk_direct_turn")

    required_capabilities = _profile_capabilities(
        profile,
        "required_capabilities",
    )
    forbidden_capabilities = _profile_capabilities(
        profile,
        "forbidden_capabilities",
    )
    if distress >= 0.30:
        required_capabilities.append("emotional_acknowledgement")
    if crisis_signal:
        required_capabilities.append("crisis_escalation")
    if urgent_health_signal:
        required_capabilities.append("urgent_medical_escalation")
    if primary_intent == "problem_solving":
        required_capabilities.extend(["actionable_solution", "reasoning"])
    if (
        (primary_intent == "decision_support" and not prediction_request)
        or _COMPARISON_RE.search(text)
    ):
        required_capabilities.append("comparison")
    if _EXPLICIT_STEPS_RE.search(text):
        required_capabilities.append("steps")
    if ambiguity >= 0.72 or clarification["required"]:
        required_capabilities.append("clarification")
    if factuality >= 0.34 and not (crisis_signal or urgent_health_signal):
        required_capabilities.append("evidence_or_calibration")
    if quantitative_request:
        required_capabilities.extend(("verified_calculation", "reasoning"))
    if unsupported_science_calculation:
        required_capabilities.append("unsupported_science_boundary")
    if investigation_request:
        required_capabilities.append("scientific_reasoning")
    if prediction_request:
        required_capabilities.extend(("calibrated_prediction", "assumptions"))
    if causal_request:
        required_capabilities.extend(("causal_reasoning", "assumptions"))
    if conversation_request:
        required_capabilities.append("conversation_continuity")
    if sycophancy >= 0.28:
        required_capabilities.append("independent_assessment")
    # The structured prompt profile is polarity-aware, while the legacy cue
    # regexes above are intentionally broad and can see a negated token such as
    # "do not give steps".  Do not let those fallback cues reintroduce a
    # capability the profile explicitly forbids.  Safety escalation and a
    # required clarification remain higher-authority response obligations.
    protected_capabilities = {
        "clarification",
        "crisis_escalation",
        "urgent_medical_escalation",
    }
    forbidden_capabilities = [
        capability
        for capability in dict.fromkeys(forbidden_capabilities)
        if capability not in protected_capabilities
    ]
    forbidden_capability_set = set(forbidden_capabilities)
    required_capabilities = [
        capability
        for capability in dict.fromkeys(required_capabilities)
        if capability in protected_capabilities
        or capability not in forbidden_capability_set
    ]

    emotion_cue = "neutral"
    if distress >= 0.58:
        emotion_cue = "possible_distress"
    elif affect["negative"] >= 0.28:
        emotion_cue = "possible_negative_affect"
    elif affect["positive"] >= 0.28:
        emotion_cue = "possible_positive_affect"

    risk_tier = "low"
    if crisis_signal or urgent_health_signal:
        risk_tier = "critical"
    elif epistemic_risk >= 0.68:
        risk_tier = "high"
    elif complexity >= 0.30 or distress >= 0.30 or ambiguity >= 0.72:
        risk_tier = "medium"

    prompt_diagnostics = _load_prompt_understanding_module().prompt_understanding_diagnostics(
        profile
    )
    if not isinstance(prompt_diagnostics, Mapping):
        prompt_diagnostics = {}
    return {
        "schema_version": "supermix-interaction-plan-v1",
        "version": PLANNER_VERSION,
        "prompt_profile": profile,
        "prompt_profile_diagnostics": dict(prompt_diagnostics),
        "appraisal": {
            "valence": round(affect["valence"], 3),
            "arousal": round(affect["arousal"], 3),
            "distress": round(distress, 3),
            "current_distress": round(current_distress, 3),
            "context_distress": round(history_distress, 3),
            "affect_trend": affect_trend,
            "continuity_applied": anchored_followup,
            "uncertainty": round(uncertainty, 3),
            "ambiguity": round(ambiguity, 3),
            "emotion_cue": emotion_cue,
            "needs_validation": bool(distress >= 0.30),
        },
        "intent": {
            "primary": primary_intent,
            "confidence": round(intent_confidence, 3),
            "scores": {key: round(value, 3) for key, value in scores.items()},
        },
        **strategy,
        "risk": {
            "tier": risk_tier,
            "epistemic_score": round(epistemic_risk, 3),
            "factuality_score": round(factuality, 3),
        },
        "deliberation": {
            "difficulty_score": round(complexity, 3),
            "epistemic_risk": round(epistemic_risk, 3),
            "value_of_compute": round(value_of_compute, 3),
            "suggested_reasoning_floor": suggested_floor,
            "prediction_stability_recommended": bool(suggested_floor >= 3),
            "reasons": deliberation_reasons,
        },
        "compute_advice": {
            "role": "shadow_advisory_only",
            "activation_available": False,
            "suggested_reasoning_floor": suggested_floor,
            "decision_exit_authority": "checkpoint_bound_prediction_verifier",
        },
        "response_contract": {
            "required_capabilities": required_capabilities,
            "forbidden_capabilities": forbidden_capabilities,
            "mixed_objective": bool(
                len(required_capabilities) >= 2
            ),
            "clarification_required": bool(
                clarification["required"]
                and not (crisis_signal or urgent_health_signal)
            ),
            "clarification_reason": clarification["reason"],
            "unresolved_reference_count": clarification[
                "unresolved_reference_count"
            ],
            "hard_conflict_count": clarification["hard_conflict_count"],
        },
        "targeted_clarification": clarification["question"],
        "guards": {
            "factuality_risk": (
                "high" if factuality >= 0.68 else "medium" if factuality >= 0.34 else "low"
            ),
            "factuality_score": round(factuality, 3),
            "verification_recommended": bool(factuality >= 0.34),
            "calibrated_uncertainty": bool(
                factuality >= 0.34
                or uncertainty >= 0.48
                or prediction_request
                or scientific_request
            ),
            "prediction_request": prediction_request,
            "scientific_request": scientific_request,
            "unsupported_science_calculation": unsupported_science_calculation,
            "investigation_request": investigation_request,
            "mathematical_request": mathematical_request,
            "quantitative_request": quantitative_request,
            "causal_request": causal_request,
            "conversation_request": conversation_request,
            "multi_part_expected": multi_part_expected,
            "sycophancy_risk": (
                "high" if sycophancy >= 0.68 else "medium" if sycophancy >= 0.28 else "low"
            ),
            "sycophancy_score": round(sycophancy, 3),
            "avoid_unearned_agreement": bool(sycophancy >= 0.28),
            "crisis_signal": crisis_signal,
            "urgent_health_signal": urgent_health_signal,
            "safety_escalation_required": bool(crisis_signal or urgent_health_signal),
        },
        "ranking_weights": {
            "empathy": round(0.08 + 0.24 * distress, 3),
            "reasoning": round(
                0.10
                + (
                    0.20
                    if primary_intent
                    in {"problem_solving", "decision_support", "explanation"}
                    else 0.0
                ),
                3,
            ),
            "actionability": round(
                0.08
                + (
                    0.18
                    if primary_intent
                    in {"problem_solving", "decision_support", "emotional_support"}
                    else 0.0
                ),
                3,
            ),
            "calibration": round(
                0.06
                + 0.22 * factuality
                + (0.12 if prediction_request else 0.0)
                + (0.06 if scientific_request else 0.0),
                3,
            ),
        },
    }


def score_candidate_for_interaction(
    candidate_text: str,
    interaction_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a bounded plan-alignment score for candidate reranking."""

    text = str(candidate_text or "").strip()
    weights = dict(interaction_plan.get("ranking_weights", {}))
    appraisal = dict(interaction_plan.get("appraisal", {}))
    guards = dict(interaction_plan.get("guards", {}))
    strategy = str(
        interaction_plan.get("response_strategy", "direct_then_offer_depth")
    )
    signals = {
        "empathy": _match_score(_EMPATHY_RESPONSE_RE, text, 0.42),
        "actionability": _match_score(_ACTION_RESPONSE_RE, text, 0.22),
        "reasoning": _match_score(_REASONING_RESPONSE_RE, text, 0.18),
        "calibration": _match_score(_CALIBRATION_RESPONSE_RE, text, 0.28),
        "verification": _match_score(_VERIFY_RESPONSE_RE, text, 0.30),
        "clarification": _match_score(_CLARIFY_RESPONSE_RE, text, 0.42),
        "comparison": _match_score(_COMPARISON_RESPONSE_RE, text, 0.34),
        "steps": _match_score(_STEP_RESPONSE_RE, text, 0.30),
        "assumptions": _assumption_response_score(text),
        "science_observation": _match_score(_SCIENCE_OBSERVATION_RE, text, 0.24),
        "science_test": _match_score(_SCIENCE_TEST_RE, text, 0.24),
        "science_boundary": _match_score(_SCIENCE_BOUNDARY_RESPONSE_RE, text, 0.42),
        "prediction": _match_score(_FORECAST_STATEMENT_RE, text, 0.24),
        "forecast_basis": _forecast_basis_score(text),
        "forecast_assumption_basis": _match_score(
            _FORECAST_ASSUMPTION_BASIS_RE,
            text,
            0.24,
        ),
        "forecast_limit": _match_score(_FORECAST_LIMIT_RE, text, 0.24),
        "abstention": _match_score(_ABSTENTION_RESPONSE_RE, text, 0.30),
        "calculation_value": _match_score(_CALCULATION_VALUE_RE, text, 0.24),
        "calculation_check": _match_score(_CALCULATION_CHECK_RE, text, 0.24),
        "causal_mechanism": _match_score(_CAUSAL_MECHANISM_RE, text, 0.22),
        "causal_limit": _match_score(_CAUSAL_LIMIT_RE, text, 0.22),
        "multi_part_coverage": _multi_part_coverage_score(
            text, guards.get("multi_part_expected")
        ),
        "crisis_support": _safety_support_score(
            _CRISIS_RESPONSE_RE, text, 0.55
        ),
        "urgent_medical_support": _safety_support_score(
            _URGENT_HEALTH_RESPONSE_RE, text, 0.55
        ),
        "unearned_agreement": _unearned_agreement_score(text),
        "overclaim": _match_score(_OVERCLAIM_RE, text, 0.22),
        "dismissive": _dismissive_score(text),
    }
    signals["independent_assessment"] = max(
        signals["calibration"], signals["verification"]
    )
    # Conjunctive contracts prevent one magic word ("evidence", "units", or
    # "assuming") from standing in for a complete scientific/forecast check.
    signals["scientific_reasoning"] = min(
        signals["science_observation"], signals["science_test"]
    )
    signals["forecast_structure_present"] = min(
        signals["prediction"],
        signals["assumptions"],
        signals["forecast_basis"],
    )
    signals["calibrated_prediction"] = min(
        signals["forecast_structure_present"],
        signals["forecast_assumption_basis"],
        max(signals["forecast_limit"], signals["abstention"]),
    )
    signals["verified_calculation"] = min(
        signals["calculation_value"], signals["calculation_check"]
    )
    signals["causal_reasoning"] = min(
        signals["causal_mechanism"], signals["causal_limit"]
    )

    response_contract = dict(interaction_plan.get("response_contract", {}))
    required = list(response_contract.get("required_capabilities", ()))
    forbidden = list(response_contract.get("forbidden_capabilities", ()))
    capability_checks = {
        "emotional_acknowledgement": signals["empathy"] > 0.0,
        "actionable_solution": signals["actionability"] > 0.0,
        "reasoning": signals["reasoning"] > 0.0,
        "comparison": signals["comparison"] > 0.0,
        "steps": signals["steps"] > 0.0,
        "clarification": signals["clarification"] > 0.0,
        "evidence_or_calibration": bool(
            signals["verification"] > 0.0 or signals["calibration"] > 0.0
        ),
        "independent_assessment": signals["independent_assessment"] > 0.0,
        "crisis_escalation": signals["crisis_support"] > 0.0,
        "urgent_medical_escalation": signals["urgent_medical_support"] > 0.0,
        "assumptions": signals["assumptions"] > 0.0,
        "scientific_reasoning": signals["scientific_reasoning"] > 0.0,
        "unsupported_science_boundary": signals["science_boundary"] > 0.0,
        "calibrated_prediction": signals["calibrated_prediction"] > 0.0,
        "verified_calculation": signals["verified_calculation"] > 0.0,
        "causal_reasoning": signals["causal_reasoning"] > 0.0,
        "multi_part_coverage": signals["multi_part_coverage"] > 0.0,
    }
    checkable_required = [
        item for item in required if item in capability_checks
    ]
    unchecked_required = [
        item for item in required if item not in capability_checks
    ]
    coverage = (
        1.0
        if not checkable_required
        else sum(
            bool(capability_checks.get(item, False))
            for item in checkable_required
        )
        / float(len(checkable_required))
    )
    forbidden_violations = [
        item
        for item in forbidden
        if item in capability_checks
        and bool(capability_checks.get(item, False))
    ]
    constraint_audit = _profile_constraint_audit(
        text,
        "",
        interaction_plan.get("prompt_profile"),
    )
    constraint_violations = constraint_audit.get("violations", ())
    if not isinstance(constraint_violations, (list, tuple)):
        constraint_violations = ()
    try:
        constraint_coverage = _clamp(
            float(constraint_audit.get("coverage", 1.0))
        )
    except (TypeError, ValueError, OverflowError):
        constraint_coverage = 1.0
    signals["contract_coverage"] = coverage
    signals["constraint_coverage"] = constraint_coverage
    signals["constraint_violation_count"] = float(len(constraint_violations))
    signals["forbidden_capability_violation_count"] = float(
        len(forbidden_violations)
    )

    total = (
        float(weights.get("empathy", 0.08)) * signals["empathy"]
        + float(weights.get("actionability", 0.08)) * signals["actionability"]
        + float(weights.get("reasoning", 0.10)) * signals["reasoning"]
        + float(weights.get("calibration", 0.06)) * signals["calibration"]
        + (0.03 * coverage if required else 0.0)
    )
    if guards.get("verification_recommended"):
        total += 0.10 * signals["verification"]
    if guards.get("prediction_request"):
        total += 0.12 * signals["calibrated_prediction"]
        total += 0.05 * signals["assumptions"]
    if guards.get("investigation_request"):
        total += 0.09 * signals["scientific_reasoning"]
        total += 0.04 * signals["assumptions"]
    if guards.get("quantitative_request"):
        total += 0.09 * signals["verified_calculation"]
    if guards.get("causal_request"):
        total += 0.08 * signals["causal_reasoning"]
    if signals["multi_part_coverage"] > 0.0:
        total += 0.08 * signals["multi_part_coverage"]
    if guards.get("avoid_unearned_agreement"):
        total -= 0.32 * signals["unearned_agreement"]
    if guards.get("calibrated_uncertainty") and not _SAFETY_CAUTION_RE.search(text):
        total -= 0.14 * signals["overclaim"]
    if float(appraisal.get("distress", 0.0)) >= 0.30:
        total -= 0.28 * signals["dismissive"]
    if strategy == "clarify_then_act":
        total += 0.18 * signals["clarification"]
    contract_penalty = min(
        0.28,
        0.08 * len(forbidden_violations)
        + 0.07 * len(constraint_violations)
        + 0.08 * (1.0 - constraint_coverage),
    )
    total -= contract_penalty
    return {
        "total": round(_clamp(total, -1.0, 1.0), 6),
        "signals": {key: round(value, 4) for key, value in signals.items()},
        "constraint_audit": constraint_audit,
        "forbidden_capability_violations": forbidden_violations,
        "unchecked_required_capabilities": unchecked_required,
        "contract_penalty": round(contract_penalty, 6),
    }


def evaluate_response_contract(
    response_text: str,
    user_text: str,
    interaction_plan: Mapping[str, Any],
    relevance_context: str = "",
) -> Dict[str, Any]:
    """Audit response obligations, including supported deterministic answers."""

    scored = score_candidate_for_interaction(response_text, interaction_plan)
    signals = dict(scored["signals"])
    response_contract = dict(interaction_plan.get("response_contract", {}))
    required = list(response_contract.get("required_capabilities", ()))
    calculation_match: Optional[bool] = None
    supported_result = _supported_calculation_result(user_text)
    empirical_estimate_check = bool(
        isinstance(supported_result, Mapping)
        and supported_result.get("problem_class") == "prediction"
        and supported_result.get("method") == "empirical_bernoulli_plugin"
    )
    calculation_required = bool(
        "verified_calculation" in required or empirical_estimate_check
    )
    if calculation_required:
        calculation_match = _candidate_matches_verified_calculation(
            response_text,
            user_text,
        )
        if calculation_match is True:
            if "verified_calculation" in required:
                signals["verified_calculation"] = 1.0
            # A matching answer here has been recomputed by the bounded local
            # verifier.  Treat that deterministic check as evidence for the
            # response contract instead of requiring a lexical magic word.
            signals["verification"] = 1.0
            if "actionable_solution" in required:
                signals["actionability"] = 1.0
        else:
            signals["verified_calculation"] = 0.0
            if empirical_estimate_check:
                signals["calibrated_prediction"] = 0.0
    signal_for_capability = {
        "emotional_acknowledgement": "empathy",
        "actionable_solution": "actionability",
        "reasoning": "reasoning",
        "comparison": "comparison",
        "steps": "steps",
        "clarification": "clarification",
        "independent_assessment": "independent_assessment",
        "crisis_escalation": "crisis_support",
        "urgent_medical_escalation": "urgent_medical_support",
        "assumptions": "assumptions",
        "scientific_reasoning": "scientific_reasoning",
        "unsupported_science_boundary": "science_boundary",
        "calibrated_prediction": "calibrated_prediction",
        "verified_calculation": "verified_calculation",
        "causal_reasoning": "causal_reasoning",
        "multi_part_coverage": "multi_part_coverage",
    }
    met = []
    missing = []
    unchecked_required = []
    for capability in required:
        if capability == "evidence_or_calibration":
            satisfied = bool(
                float(signals.get("verification", 0.0)) > 0.0
                or float(signals.get("calibration", 0.0)) > 0.0
            )
        else:
            signal = signal_for_capability.get(capability)
            if signal is None:
                unchecked_required.append(capability)
                continue
            satisfied = bool(float(signals.get(signal, 0.0)) > 0.0)
        (met if satisfied else missing).append(capability)

    guards = dict(interaction_plan.get("guards", {}))
    violations = []
    if calculation_match is False:
        violations.append(
            "prediction_estimate_mismatch"
            if empirical_estimate_check
            else "calculation_mismatch"
        )
    elif calculation_match is None and calculation_required:
        violations.append("calculation_not_verifiable")
    if (
        guards.get("unsupported_science_calculation")
        and _unsupported_science_answer_asserted(response_text, user_text)
    ):
        violations.append("unsupported_science_answer_asserted")
    if (
        guards.get("avoid_unearned_agreement")
        and float(signals.get("unearned_agreement", 0.0)) > 0.0
    ):
        violations.append("unearned_agreement")
    if (
        guards.get("calibrated_uncertainty")
        and float(signals.get("overclaim", 0.0)) > 0.0
        and not _SAFETY_CAUTION_RE.search(str(response_text or ""))
    ):
        violations.append("unsupported_certainty")
    if float(signals.get("dismissive", 0.0)) > 0.0:
        violations.append("dismissive_language")
    forbidden_capability_violations = list(
        scored.get("forbidden_capability_violations", ())
    )
    violations.extend(
        f"forbidden_capability:{capability}"
        for capability in forbidden_capability_violations
    )
    constraint_audit = _profile_constraint_audit(
        response_text,
        user_text,
        interaction_plan.get("prompt_profile"),
    )
    raw_constraint_violations = constraint_audit.get("violations", ())
    if not isinstance(raw_constraint_violations, (list, tuple)):
        raw_constraint_violations = ()
    for finding in raw_constraint_violations:
        if isinstance(finding, Mapping):
            finding_id = str(
                finding.get("constraint_id")
                or finding.get("kind")
                or "deterministic"
            )
        else:
            finding_id = str(finding or "deterministic")
        violations.append(f"constraint:{finding_id}")
    violations = list(dict.fromkeys(violations))
    checked_required_count = len(met) + len(missing)
    coverage = (
        1.0
        if not checked_required_count
        else len(met) / float(checked_required_count)
    )
    return {
        "accepted": bool(not missing and not violations),
        "coverage": round(coverage, 4),
        "required": required,
        "met": met,
        "missing": missing,
        "unchecked_required_capabilities": unchecked_required,
        "violations": violations,
        "forbidden_capability_violations": forbidden_capability_violations,
        "constraint_audit": constraint_audit,
        "lexical_relevance": round(
            _lexical_relevance(
                " ".join(
                    part
                    for part in (
                        str(relevance_context or ""),
                        str(user_text or ""),
                    )
                    if part
                ),
                response_text,
            ),
            4,
        ),
    }


def finalize_response_for_interaction(
    response_text: str,
    user_text: str,
    interaction_plan: Mapping[str, Any],
    relevance_context: str = "",
) -> Dict[str, Any]:
    """Apply conservative repairs when a selected response violates the plan."""

    raw = str(response_text or "").strip()
    guards = dict(interaction_plan.get("guards", {}))
    signals = score_candidate_for_interaction(raw, interaction_plan)["signals"]
    initial = evaluate_response_contract(
        raw, user_text, interaction_plan, relevance_context
    )

    def result(text: str, changed: bool, reason: str) -> Dict[str, Any]:
        return {
            "text": text,
            "changed": changed,
            "reason": reason,
            "audit": evaluate_response_contract(
                text, user_text, interaction_plan, relevance_context
            ),
        }

    if guards.get("crisis_signal"):
        if float(signals.get("crisis_support", 0.0)) > 0.0:
            if float(signals.get("empathy", 0.0)) <= 0.0:
                return result(
                    "I'm really sorry you're facing this. " + raw,
                    True,
                    "crisis_guidance_preserved_with_acknowledgement",
                )
            return result(raw, False, "crisis_guidance_preserved")
        return result(
            "I'm really sorry you're facing this, and your immediate safety matters. "
            "If you might act now or are in immediate danger, contact local emergency "
            "services now or go to the nearest emergency department. If you can, "
            "contact a crisis service and a trusted person who can stay with you, "
            "and move away from anything you could use to hurt yourself. "
            "Are you in immediate danger right now?",
            True,
            "crisis_safety_escalation",
        )

    if guards.get("urgent_health_signal"):
        if float(signals.get("urgent_medical_support", 0.0)) > 0.0:
            return result(raw, False, "urgent_medical_guidance_preserved")
        return result(
            "These symptoms can require urgent medical assessment. Contact local "
            "emergency services now or seek immediate emergency care, especially "
            "if the symptoms are severe, new, or worsening.",
            True,
            "urgent_medical_safety_escalation",
        )

    response_contract = dict(interaction_plan.get("response_contract", {}))
    if response_contract.get("clarification_required"):
        clarification = str(
            interaction_plan.get("targeted_clarification")
            or "Which specific result do you want me to produce?"
        ).strip()
        if not clarification.endswith("?"):
            clarification = clarification.rstrip(".! ") + "?"
        return result(
            clarification,
            clarification != raw,
            str(
                response_contract.get("clarification_reason")
                or "targeted_clarification"
            ),
        )

    if (
        guards.get("avoid_unearned_agreement")
        and float(signals.get("unearned_agreement", 0.0)) > 0.0
    ):
        return result(
            "I can help assess that, but I should not confirm it without checking "
            "the evidence. Share the exact claim or source, and I will separate "
            "what is supported from what remains uncertain.",
            True,
            "unearned_agreement_blocked",
        )
    if "dismissive_language" in initial.get("violations", ()):
        return result(
            "That sounds difficult, and I don't want to dismiss what you're dealing "
            "with. We can slow it down and identify one manageable next step together.",
            True,
            "dismissive_language_blocked",
        )
    if guards.get("unsupported_science_calculation") and (
        "unsupported_science_boundary" in initial.get("missing", ())
        or "unsupported_science_answer_asserted" in initial.get("violations", ())
    ):
        fallback = (
            "I can't safely verify a numeric result for this request with the bounded "
            "scientific models because the required assumptions or authority are not "
            "established. The next step is to provide an explicit supported model for "
            "a non-high-stakes exercise, or consult a qualified domain professional."
        )
        return result(
            fallback,
            fallback != raw,
            "unsupported_science_calculation_blocked",
        )
    if _incompatible_arithmetic_template(
        raw,
        user_text,
        interaction_plan,
        relevance_context,
    ):
        fallback = (
            "I don't have enough relevant information to answer that reliably."
        )
        repair_api = _load_prompt_understanding_module()
        repair_fn = getattr(repair_api, "repair_response_constraints", None)
        repair = (
            repair_fn(
                fallback,
                user_text,
                interaction_plan.get("prompt_profile"),
            )
            if callable(repair_fn)
            else {"text": fallback, "changed": False}
        )
        return result(
            str(repair.get("text") or fallback),
            True,
            "incompatible_arithmetic_template_blocked",
        )
    repair_api = _load_prompt_understanding_module()
    repair_fn = getattr(repair_api, "repair_response_constraints", None)
    repair = (
        repair_fn(
            raw,
            user_text,
            interaction_plan.get("prompt_profile"),
        )
        if callable(repair_fn)
        else {"text": raw, "changed": False}
    )
    if bool(repair.get("changed", False)):
        return result(
            str(repair.get("text") or raw),
            True,
            "deterministic_constraints_repaired",
        )
    # Lower-precision findings (missing empathy, topical continuity, unsupported
    # certainty, and lexical relevance) remain audit/ranking signals in v2.
    # Rewriting on those heuristics would risk replacing semantically relevant
    # answers that happen to use synonyms absent from the lexical matcher.
    return result(
        raw,
        False,
        "candidate_aligned"
        if initial["accepted"]
        else "candidate_partially_aligned",
    )


def interaction_plan_diagnostics(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Return stable, compact diagnostics suitable for APIs and traces."""

    appraisal = dict(plan.get("appraisal", {}))
    intent = dict(plan.get("intent", {}))
    guards = dict(plan.get("guards", {}))
    deliberation = dict(plan.get("deliberation", {}))
    response_contract = dict(plan.get("response_contract", {}))
    risk = dict(plan.get("risk", {}))
    return {
        "schema_version": plan.get("schema_version"),
        "version": plan.get("version"),
        "intent": intent.get("primary", "conversation"),
        "intent_confidence": intent.get("confidence", 0.0),
        "strategy": plan.get("response_strategy", "direct_then_offer_depth"),
        "reasoning_mode": plan.get("reasoning_mode", "direct"),
        "risk_tier": risk.get("tier", "low"),
        "emotion_cue": appraisal.get("emotion_cue", "neutral"),
        "distress": appraisal.get("distress", 0.0),
        "context_distress": appraisal.get("context_distress", 0.0),
        "affect_trend": appraisal.get(
            "affect_trend", "insufficient_context"
        ),
        "continuity_applied": bool(
            appraisal.get("continuity_applied", False)
        ),
        "uncertainty": appraisal.get("uncertainty", 0.0),
        "ambiguity": appraisal.get("ambiguity", 0.0),
        "factuality_risk": guards.get("factuality_risk", "low"),
        "verification_recommended": bool(
            guards.get("verification_recommended", False)
        ),
        "sycophancy_risk": guards.get("sycophancy_risk", "low"),
        "safety_escalation_required": bool(
            guards.get("safety_escalation_required", False)
        ),
        "deliberation": deliberation,
        "compute_advice": dict(plan.get("compute_advice", {})),
        "response_contract": response_contract,
        "prompt_understanding": dict(
            plan.get("prompt_profile_diagnostics", {})
        ),
    }
