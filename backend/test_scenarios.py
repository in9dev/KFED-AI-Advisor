"""
test_scenarios.py — runs the full multi-agent pipeline end-to-end against a
handful of realistic bilingual test scenarios (standing in for the KFED
"Test Scenarios" resource mentioned in the challenge brief, until the team
receives the official ones from mentors/the KFED expert).

Run with:  python3 -m backend.test_scenarios
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import Orchestrator  # noqa: E402
from backend import store  # noqa: E402

SCENARIOS = [
    {
        "label": "New Emirati entrepreneur, agri idea, English",
        "beneficiary_id": "GUEST-1",
        "message": "Hi, I have a greenhouse farming idea in Al Ain but I have no funding and don't know where to start.",
    },
    {
        "label": "Returning beneficiary with scattered KFED history (Arabic)",
        "beneficiary_id": "B-1042",
        "message": "أريد المساعدة في تصدير منتجاتي الزراعية إلى أسواق جديدة، أحتاج أيضًا تمويل توسع.",
    },
    {
        "label": "Growth-stage manufacturer needing procurement + ICV",
        "beneficiary_id": "B-2077",
        "message": "My factory is ready to bid on government contracts, what do I need?",
    },
    {
        "label": "Escalation trigger: large funding ask",
        "beneficiary_id": "GUEST-2",
        "message": "I need a loan of about 8 million AED to build a new manufacturing plant.",
    },
    {
        "label": "Escalation trigger: distress / explicit human request",
        "beneficiary_id": "GUEST-3",
        "message": "أنا محبط جدًا ولم أعد أثق بالنظام، أريد التحدث مع مستشار بشري حقيقي.",
    },
    {
        "label": "Youth gaming interest",
        "beneficiary_id": "GUEST-4",
        "message": "I'm 20 years old and I want to build my own video game studio.",
    },
]


QUIZ_SCENARIOS = [
    {
        "label": "Quiz path: agri idea, Arabic",
        "beneficiary_id": "QUIZ-1",
        "language": "ar",
        "answers": {
            "business_name": "مزرعة الواحة",
            "sector": "agri",
            "stage": "idea",
            "skill_gaps": ["funding"],
            "context": "عندي بيت محمي صغير وأحتاج تمويل توسع.",
        },
    },
    {
        "label": "Quiz path: large funding ask triggers escalation",
        "beneficiary_id": "QUIZ-2",
        "language": "en",
        "answers": {
            "business_name": "Falcon Steel Works",
            "sector": "manufacturing",
            "stage": "growth",
            "skill_gaps": ["procurement"],
            "context": "We need a loan of 9 million AED to build a new plant.",
        },
    },
]


def run_quiz_scenarios(orch):
    failures = []
    for sc in QUIZ_SCENARIOS:
        print("=" * 78)
        print("QUIZ SCENARIO:", sc["label"])
        bid = sc["beneficiary_id"]
        orch.restart_quiz(bid)
        step = orch.start_quiz(bid, sc["language"])
        result = None
        for qid, answer in sc["answers"].items():
            assert step["question"]["question_id"] == qid, \
                f"expected question {qid}, backend asked for {step['question']['question_id']}"
            step = orch.submit_quiz_answer(bid, qid, answer, sc["language"])
        result = step
        print("REPLY:\n" + (result.get("reply") or "(none)"))
        if result.get("escalation"):
            print("ESCALATED:", result["escalation"]["reason_code"])
        print("PATHWAY ITEMS:", len(result.get("pathway", [])))

        if not result.get("quiz_complete"):
            failures.append(sc["label"] + ": quiz did not complete")
        if not result.get("reply"):
            failures.append(sc["label"] + ": empty reply after quiz completion")
        if "Escalation" in sc["label"] or "escalation" in sc["label"].lower():
            if not result.get("escalation"):
                failures.append(sc["label"] + ": expected an escalation but none was raised")
    return failures


def run():
    store.reset_state()
    orch = Orchestrator()
    failures = run_quiz_scenarios(orch)

    for sc in SCENARIOS:
        print("=" * 78)
        print("SCENARIO:", sc["label"])
        print("INPUT   :", sc["message"])
        result = orch.handle_turn(sc["beneficiary_id"], sc["message"])
        print("LANGUAGE:", result["language"])
        print("REPLY   :\n" + result["reply"])
        print("RETRIEVED:", [r["name_en"] for r in result["retrieved"]])
        print("PATHWAY ITEMS:", len(result["pathway"]))
        if result["escalation"]:
            print("ESCALATED:", result["escalation"]["reason_code"], "-", result["escalation"]["reason_text"])
        if result["alerts"]:
            print("ALERTS:", [a["en"] for a in result["alerts"]])

        # sanity checks — this is the "it actually works, not placeholder text" check
        if not result["reply"] or len(result["reply"]) < 10:
            failures.append(sc["label"] + ": empty/short reply")
        if sc["label"].startswith("Escalation") and not result["escalation"]:
            failures.append(sc["label"] + ": expected an escalation but none was raised")

    print("=" * 78)
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        total = len(SCENARIOS) + len(QUIZ_SCENARIOS)
        print(f"All {total} scenarios ({len(QUIZ_SCENARIOS)} quiz-path + {len(SCENARIOS)} chat-path) "
              f"ran end-to-end with no placeholder/empty output.")


if __name__ == "__main__":
    run()
