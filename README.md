# KFED AI Entrepreneur Advisor

**Team Falcon Systems** — Nasser Almarbooei, Ahmed AlAhmed, Mohammed Alkaabi, Falah Alhosani, Rashed Alhosani.

Built for the **Agentic Verse AI Program** (ADEK × 42 Abu Dhabi), Challenge 1,
in partnership with the **Khalifa Fund for Enterprise Development (KFED)**.

A multi-agent AI system that profiles a beneficiary through a short onboarding
quiz, recommends a personalised KFED programme pathway, tracks their progress,
and escalates complex cases to a human advisor — bilingually, in Arabic and
English.

## The flow

The entrepreneur never lands cold in a chatbox. They go through five steps:

1. **Profile (the quiz)** — a short, structured onboarding quiz (business name,
   sector, stage, skill gaps, free-text context) drives the ProfilerAgent
   deterministically, instead of guessing from free text.
2. **Recommend** — RAG retrieval + the RecommenderAgent turn that profile into
   a Now/Next/Later pathway grounded in real KFED programmes.
3. **Track** — the ProgressTrackerAgent persists milestones and proactively
   checks real programme deadlines.
4. **Escalate** — the EscalationAgent scans the quiz's free-text answers (and
   any later chat) for complexity and hands off to a human KFED advisor.
5. **Two languages, throughout** — the quiz, the recommendations, the tracker
   alerts, and the escalation summaries all work natively in Arabic and English.

After the quiz completes, the same chat box is available for follow-ups
(progress updates, further questions) — see `backend/agents.py`'s
`Orchestrator.handle_turn` for that path, and `Orchestrator.start_quiz` /
`submit_quiz_answer` for the profiling flow.

## Run it (zero installs)

```bash
cd kfed-ai-advisor
python3 -m backend.app
# open http://localhost:8420
```

No pip installs required — the backend uses only the Python standard library,
so it works on any machine with Python 3.8+, including offline at a venue.

Verify everything end-to-end (also doubles as a regression test / demo backup):

```bash
python3 -m backend.test_scenarios
```

## How the must-have AI capabilities map to this build

| Requirement | Where |
|---|---|
| **Autonomous agent** | `ProgressTrackerAgent.check_alerts()` and `EscalationAgent.check()` act on their own initiative every turn — checking real KFED programme dates and scanning for risk — without being explicitly asked. |
| **RAG** | `backend/rag.py` — a from-scratch TF-IDF + cosine-similarity index over `kb/programs.json`, which is real KFED programme data pulled from khalifafund.ae (funding products, SME Champions, ICV Readiness, Light Manufacturing Accelerator, MZN Hub71, Ruwad Al Ain Bootcamp, the Entrepreneurship Competition, etc.) — not generic LLM knowledge. |
| **Multi-agent pipeline** | Five agents with distinct jobs: `ProfilerAgent` (now driven by the onboarding quiz via `ProfilerAgent.from_quiz()`) → `EscalationAgent` → `RetrieverAgent` → `RecommenderAgent` → `ProgressTrackerAgent`, coordinated by `Orchestrator.submit_quiz_answer()` (profiling) and `Orchestrator.handle_turn()` (post-quiz chat) in `backend/agents.py`. |
| **Human-in-the-loop** | `EscalationAgent` triggers on: funding asks beyond KFED's standard loan ceilings, legal/dispute language, beneficiary distress, explicit requests for a human, or a knowledge-base coverage gap — and produces a structured handoff summary (see `n8n/` for how that would page a real advisor). |

## Project layout

```
kb/programs.json          Real KFED programme knowledge base (bilingual), incl. key dates
data/beneficiaries_seed.json   Simulated "scattered systems": CRM + training history + coaching notes
backend/rag.py             TF-IDF retrieval engine (the RAG layer)
backend/agents.py          The 5 agents + Orchestrator
backend/llm.py             Single swap-point to plug in a real LLM (Anthropic/OpenAI) later
backend/store.py           JSON-file persistence (stands in for Qudorat / real KFED data stores)
backend/app.py             HTTP server (stdlib only) + API
backend/test_scenarios.py  End-to-end scenario tests (EN + AR, incl. escalation cases)
frontend/index.html        Bilingual single-page chat UI (EN/AR, RTL-aware)
n8n/                        Bonus: importable n8n workflow fronting the same backend
```

## Why there's no LLM wired in yet

No API key was available while building this. Every "generated" sentence you
see is composed from a template that's *filled with data retrieved from the
real KFED knowledge base* — so nothing in the demo is a hallucination, and the
personalisation is genuinely driven by the profile + retrieval, not canned
text (see `RecommenderAgent.compose_message`, and run `test_scenarios.py` to
see six different profiles produce six different pathways).

To go fully live: add an Anthropic API key in `backend/llm.py` (one file,
clearly marked) and each agent's template-composition step can be replaced by
a real generation call — the retrieval, profiling, escalation, and tracking
logic underneath does not need to change.

## Known gaps / next steps before Day 5

- Swap the keyword-based profiler for the official KFED "Test Scenarios" once
  received from mentors/the KFED expert — current scenarios in
  `test_scenarios.py` are realistic stand-ins built from public KFED program
  pages, not the official set.
- Wire a real LLM (see above) for more natural bilingual phrasing.
- Replace `data/beneficiaries_seed.json` with a real (anonymised) KFED extract
  if the team gets access, to make the "scattered data unification" demo land
  even harder with the judges.
