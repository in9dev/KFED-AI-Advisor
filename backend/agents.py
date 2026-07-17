"""
agents.py — the multi-agent pipeline for the KFED AI Entrepreneur Advisor.

Five agents, each with a distinct job, coordinated by an Orchestrator:

  1. ProfilerAgent      — unifies scattered KFED records + free-text intake
                           into one structured beneficiary profile.
  2. EscalationAgent     — autonomously watches every turn for complexity
                           triggers and hands off to a human KFED advisor
                           with a structured case summary (human-in-the-loop).
  3. RetrieverAgent      — wraps rag.KnowledgeBase; does the actual RAG
                           lookup against real KFED programme data.
  4. RecommenderAgent    — turns retrieved programmes + profile into a
                           sequenced, personalised pathway (Now / Next / Later).
  5. ProgressTrackerAgent— persists pathway + milestones, and *on its own*
                           (no user prompt needed) checks KFED programme
                           key dates against today and raises alerts.

ProfilerAgent + RetrieverAgent + RecommenderAgent + ProgressTrackerAgent make
up the multi-agent pipeline requirement. ProgressTrackerAgent and
EscalationAgent are the autonomous pieces: they act on their own initiative
(checking dates, scanning for risk) rather than only responding to a direct
question.
"""

import copy
import datetime
import re

from . import llm, store
from .rag import KnowledgeBase

TODAY = datetime.date(2026, 7, 16)  # matches the session's current date

# ---------------------------------------------------------------------------
# Bilingual keyword dictionaries (lightweight NLU — no external NLP deps)
# ---------------------------------------------------------------------------

SECTOR_KEYWORDS = {
    "agri": ["farm", "agri", "greenhouse", "crop", "livestock", "زراع", "مزرعة", "محاصيل", "بيت محمي"],
    "manufacturing": ["manufactur", "factory", "production line", "تصنيع", "مصنع", "إنتاج"],
    "tech": ["software", "app", "saas", "startup tech", "platform", "ai", "تقنية", "تطبيق", "برمجة", "ذكاء اصطناعي"],
    "gaming": ["game", "gaming", "لعبة", "ألعاب"],
    "tourism": ["tourism", "hospitality", "tour", "سياح", "ضيافة"],
    "ecommerce": ["e-commerce", "ecommerce", "online store", "تجارة إلكترونية", "متجر إلكتروني"],
    "retail": ["retail", "shop", "store", "تجزئة", "محل"],
    "creative": ["design", "media", "content", "إبداع", "تصميم", "محتوى"],
}

STAGE_KEYWORDS = {
    "youth": ["age 12", "age 15", "school student", "طالب مدرسة", "12 سنة", "15 سنة"],
    "idea": ["just an idea", "thinking of starting", "haven't started", "فكرة فقط", "لم أبدأ بعد", "أفكر في البدء"],
    "startup": ["prototype", "mvp", "just launched", "started this year", "نموذج أولي", "أطلقت مؤخرا", "بدأت هذا العام"],
    "growth": ["running for", "years in business", "revenue", "expanding", "scale", "توسع", "نمو", "إيرادات", "سنوات في السوق"],
}

SKILL_GAP_KEYWORDS = {
    "funding": ["funding", "loan", "capital", "تمويل", "قرض", "رأس مال"],
    "marketing": ["marketing", "customers", "sales", "تسويق", "عملاء", "مبيعات"],
    "export": ["export", "international market", "تصدير", "أسواق دولية"],
    "digital": ["digital transformation", "website", "رقمنة", "موقع إلكتروني"],
    "ai_skills": ["ai skills", "prompt engineering", "مهارات ذكاء اصطناعي", "هندسة الأوامر"],
    "procurement": ["government contract", "procurement", "tender", "مناقصة", "مشتريات حكومية"],
    "licensing": ["trade license", "icv certificate", "رخصة تجارية", "شهادة القيمة المضافة"],
}

ESCALATION_PATTERNS = {
    "large_funding": [
        r"\b([4-9]|[1-9]\d)\s*(million|m)\b", r"AED\s*[4-9]\d{6,}",
        r"(\d+)\s*مليون",
    ],
    "legal_or_dispute": [
        "lawsuit", "sue", "dispute", "bankrupt", "fraud", "قضية", "نزاع", "إفلاس", "احتيال",
    ],
    "distress": [
        "very upset", "angry", "furious", "lost everything", "give up", "frustrated",
        "غاضب", "محبط", "فقدت كل شيء", "لم أعد أثق", "أستسلم",
    ],
    "explicit_human_request": [
        "talk to a human", "speak to an advisor", "real person", "human advisor",
        "أريد التحدث مع مستشار", "أتحدث مع شخص حقيقي", "موظف بشري",
    ],
}


# ---------------------------------------------------------------------------
# Structured onboarding quiz (the profiling process)
# ---------------------------------------------------------------------------
# Instead of relying purely on free-text keyword-spotting, the beneficiary is
# walked through a short bilingual quiz first. Answers map deterministically
# onto the same profile fields ProfilerAgent has always used, so everything
# downstream (RAG, Recommender, Tracker, Escalation) is unchanged.

QUIZ_QUESTIONS = [
    {
        "id": "business_name",
        "type": "text",
        "optional": False,
        "prompt": {"en": "Let's start with your business. What's it called (or what's your idea)?",
                   "ar": "لنبدأ بعملك. ما اسمه؟ (أو ما هي فكرتك؟)"},
    },
    {
        "id": "sector",
        "type": "choice",
        "optional": False,
        "prompt": {"en": "Which best describes your sector?", "ar": "أي قطاع يصف عملك بشكل أفضل؟"},
        "options": [
            {"value": "agri", "label": {"en": "Agriculture", "ar": "الزراعة"}},
            {"value": "manufacturing", "label": {"en": "Manufacturing", "ar": "التصنيع"}},
            {"value": "tech", "label": {"en": "Technology / Software", "ar": "التقنية / البرمجيات"}},
            {"value": "gaming", "label": {"en": "Gaming", "ar": "الألعاب"}},
            {"value": "tourism", "label": {"en": "Tourism & Hospitality", "ar": "السياحة والضيافة"}},
            {"value": "ecommerce", "label": {"en": "E-commerce", "ar": "التجارة الإلكترونية"}},
            {"value": "retail", "label": {"en": "Retail", "ar": "التجزئة"}},
            {"value": "creative", "label": {"en": "Creative / Media / Design", "ar": "الإبداع / الإعلام / التصميم"}},
            {"value": "general", "label": {"en": "Something else", "ar": "شيء آخر"}},
        ],
    },
    {
        "id": "stage",
        "type": "choice",
        "optional": False,
        "prompt": {"en": "What stage is your business at?", "ar": "في أي مرحلة يقع عملك؟"},
        "options": [
            {"value": "youth", "label": {"en": "I'm a student exploring entrepreneurship",
                                          "ar": "طالب أستكشف ريادة الأعمال"}},
            {"value": "idea", "label": {"en": "Just an idea — haven't started yet",
                                         "ar": "مجرد فكرة — لم أبدأ بعد"}},
            {"value": "startup", "label": {"en": "I have a prototype / MVP, or just launched",
                                            "ar": "لدي نموذج أولي، أو أطلقت مؤخرًا"}},
            {"value": "growth", "label": {"en": "Running with revenue, looking to expand",
                                           "ar": "أعمل بإيرادات وأتطلع للتوسع"}},
        ],
    },
    {
        "id": "skill_gaps",
        "type": "multichoice",
        "optional": False,
        "prompt": {"en": "What do you need help with most? (choose all that apply)",
                   "ar": "ما الذي تحتاج مساعدة فيه أكثر؟ (اختر كل ما ينطبق)"},
        "options": [
            {"value": "funding", "label": {"en": "Funding / loans", "ar": "التمويل / القروض"}},
            {"value": "marketing", "label": {"en": "Marketing & customers", "ar": "التسويق والعملاء"}},
            {"value": "export", "label": {"en": "Export / international markets", "ar": "التصدير / الأسواق الدولية"}},
            {"value": "digital", "label": {"en": "Digital transformation", "ar": "التحول الرقمي"}},
            {"value": "ai_skills", "label": {"en": "AI skills", "ar": "مهارات الذكاء الاصطناعي"}},
            {"value": "procurement", "label": {"en": "Government contracts", "ar": "العقود الحكومية"}},
            {"value": "licensing", "label": {"en": "Licensing / ICV certificate", "ar": "الترخيص / شهادة القيمة المضافة"}},
        ],
    },
    {
        "id": "context",
        "type": "text",
        "optional": True,
        "prompt": {"en": "Anything else you'd like your advisor to know?",
                   "ar": "هل تود إخبار المستشار بأي شيء آخر؟"},
    },
]

_QUIZ_QUESTIONS_BY_ID = {q["id"]: q for q in QUIZ_QUESTIONS}


def localize_question(question, language):
    lang = language if language in ("en", "ar") else "en"
    payload = {
        "question_id": question["id"],
        "type": question["type"],
        "optional": question.get("optional", False),
        "prompt": question["prompt"][lang],
    }
    if "options" in question:
        payload["options"] = [{"value": o["value"], "label": o["label"][lang]} for o in question["options"]]
    return payload


def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r"[ء-ي]", text or ""))
    latin_chars = len(re.findall(r"[A-Za-z]", text or ""))
    return "ar" if arabic_chars > latin_chars else "en"


def _match_any(text_lower, keywords):
    return any(kw in text_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Agent 1: Profiler
# ---------------------------------------------------------------------------

class ProfilerAgent:
    """Understands who the entrepreneur is. Tools: keyword classifiers +
    the scattered-records lookup tool (store.lookup_scattered_records).
    Decides on its own which fields are still missing."""

    name = "ProfilerAgent"

    def run(self, beneficiary_id, message, profile):
        text_lower = (message or "").lower()
        profile = copy.deepcopy(profile) if profile else {
            "beneficiary_id": beneficiary_id,
            "language": "en",
            "sector": None,
            "stage": None,
            "skill_gaps": [],
            "history_unified": False,
            "crm": None,
            "training_history": [],
            "coaching_notes": [],
        }

        profile["language"] = detect_language(message) if message else profile.get("language", "en")

        # First-contact tool call: unify scattered KFED records.
        if not profile["history_unified"]:
            records = store.lookup_scattered_records(beneficiary_id)
            if records:
                profile["crm"] = records["crm"]
                profile["training_history"] = records["training_history"]
                profile["coaching_notes"] = records["coaching_notes"]
                if records["crm"] and not profile["sector"]:
                    profile["sector"] = records["crm"].get("declared_sector")
            profile["history_unified"] = True

        for sector, kws in SECTOR_KEYWORDS.items():
            if _match_any(text_lower, kws):
                profile["sector"] = sector
                break

        for stage, kws in STAGE_KEYWORDS.items():
            if _match_any(text_lower, kws):
                profile["stage"] = stage
                break
        if not profile.get("stage"):
            profile["stage"] = "startup"  # sensible default so pipeline still produces real output

        if not profile.get("sector"):
            profile["sector"] = "general"

        for skill, kws in SKILL_GAP_KEYWORDS.items():
            if _match_any(text_lower, kws) and skill not in profile["skill_gaps"]:
                profile["skill_gaps"].append(skill)

        profile["missing_fields"] = []
        if profile["sector"] == "general":
            profile["missing_fields"].append("sector")
        if not profile["skill_gaps"]:
            profile["missing_fields"].append("skill_gaps")

        return profile

    def from_quiz(self, beneficiary_id, answers, language):
        """Builds the profile deterministically from structured quiz answers
        instead of guessing from free text — this *is* the profiling process
        the entrepreneur walks through before anything else happens."""
        profile = {
            "beneficiary_id": beneficiary_id,
            "language": language if language in ("en", "ar") else "en",
            "business_name": (answers.get("business_name") or "").strip(),
            "sector": answers.get("sector") or "general",
            "stage": answers.get("stage") or "startup",
            "skill_gaps": list(answers.get("skill_gaps") or []),
            "context": (answers.get("context") or "").strip(),
            "history_unified": False,
            "crm": None,
            "training_history": [],
            "coaching_notes": [],
        }

        # Still unify KFED's scattered records for a returning beneficiary —
        # the quiz doesn't replace that, it just replaces free-text guessing.
        records = store.lookup_scattered_records(beneficiary_id)
        if records:
            profile["crm"] = records["crm"]
            profile["training_history"] = records["training_history"]
            profile["coaching_notes"] = records["coaching_notes"]
        profile["history_unified"] = True

        # A little extra signal from the free-text answers, merged rather than
        # overriding the beneficiary's explicit choices.
        extra_text = f"{profile['business_name']} {profile['context']}".lower()
        for skill, kws in SKILL_GAP_KEYWORDS.items():
            if _match_any(extra_text, kws) and skill not in profile["skill_gaps"]:
                profile["skill_gaps"].append(skill)

        profile["missing_fields"] = []
        return profile


# ---------------------------------------------------------------------------
# Agent 2: Escalation (human-in-the-loop)
# ---------------------------------------------------------------------------

class EscalationAgent:
    """Runs on every turn, independent of what the user asked for. Decides,
    on its own initiative, whether a case is too complex / sensitive / out of
    the KB's coverage for the AI to safely handle alone."""

    name = "EscalationAgent"

    def check(self, profile, message, retrieved_docs):
        text_lower = (message or "").lower()

        for pattern in ESCALATION_PATTERNS["large_funding"]:
            if re.search(pattern, text_lower):
                return self._case(profile, "large_funding_request",
                                   "Funding amount mentioned appears to exceed standard KFED loan"
                                   " product ceilings (Expansion Loan tops out near AED 3M) and needs"
                                   " a human credit assessment.", urgency="high")

        if _match_any(text_lower, ESCALATION_PATTERNS["legal_or_dispute"]):
            return self._case(profile, "legal_or_dispute",
                               "Message references a legal/financial dispute matter outside the"
                               " advisor's scope.", urgency="high")

        if _match_any(text_lower, ESCALATION_PATTERNS["distress"]):
            return self._case(profile, "beneficiary_distress",
                               "Language suggests the entrepreneur is frustrated or in distress;"
                               " a human advisor should follow up personally.", urgency="high")

        if _match_any(text_lower, ESCALATION_PATTERNS["explicit_human_request"]):
            return self._case(profile, "explicit_request",
                               "Beneficiary explicitly asked to speak with a human advisor.",
                               urgency="medium")

        if message and not retrieved_docs:
            return self._case(profile, "coverage_gap",
                               "No KFED knowledge-base programme matched this request with"
                               " meaningful confidence — outside current RAG coverage.",
                               urgency="low")

        return None

    @staticmethod
    def _case(profile, reason_code, reason_text, urgency):
        return {
            "reason_code": reason_code,
            "reason_text": reason_text,
            "urgency": urgency,
            "beneficiary_id": profile.get("beneficiary_id"),
            "profile_snapshot": {
                "sector": profile.get("sector"),
                "stage": profile.get("stage"),
                "skill_gaps": profile.get("skill_gaps"),
            },
            "generated_on": TODAY.isoformat(),
        }


# ---------------------------------------------------------------------------
# Agent 3: Retriever (RAG)
# ---------------------------------------------------------------------------

class RetrieverAgent:
    name = "RetrieverAgent"

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def run(self, profile, message, top_k=4):
        query = " ".join(filter(None, [
            message or "",
            profile.get("sector") or "",
            profile.get("stage") or "",
            " ".join(profile.get("skill_gaps", [])),
        ]))
        return self.kb.search(query, stage=profile.get("stage"), sector=profile.get("sector"), top_k=top_k)


# ---------------------------------------------------------------------------
# Agent 4: Recommender
# ---------------------------------------------------------------------------

_SEQUENCE_ORDER = {
    "training": 0, "bootcamp": 0, "accelerator": 1, "incubator": 1,
    "funding": 2, "certification": 2, "procurement": 3, "export": 3,
    "franchise": 3, "competition": 3, "community": 1,
}

_BUCKET_LABEL = {0: "now", 1: "now", 2: "next", 3: "later"}


class RecommenderAgent:
    name = "RecommenderAgent"

    def build(self, profile, retrieved):
        completed_ids = {h["program_id"] for h in profile.get("training_history", [])
                          if h.get("result") == "completed"}

        candidates = [(doc, score) for doc, score in retrieved if doc["id"] not in completed_ids]

        pathway = []
        for doc, score in candidates:
            bucket_rank = _SEQUENCE_ORDER.get(doc["category"], 2)
            pathway.append({
                "program_id": doc["id"],
                "name_en": doc["name_en"],
                "name_ar": doc["name_ar"],
                "category": doc["category"],
                "bucket": _BUCKET_LABEL[bucket_rank],
                "bucket_rank": bucket_rank,
                "relevance_score": score,
                "description_en": doc["description_en"],
                "description_ar": doc["description_ar"],
                "url": doc.get("url"),
                "key_dates": doc.get("key_dates"),
                "milestone_status": "recommended",
            })

        pathway.sort(key=lambda item: (item["bucket_rank"], -item["relevance_score"]))
        return pathway

    def compose_message(self, profile, pathway, escalation):
        lang = profile.get("language", "en")
        if not pathway:
            return (
                "لم أجد بعد برنامجًا مطابقًا بدقة من صندوق خليفة؛ سيتم تحويل طلبك إلى مستشار بشري."
                if lang == "ar" else
                "I couldn't confidently match a KFED programme yet — routing this to a human advisor."
            )

        top = pathway[:3]
        if lang == "ar":
            lines = ["بناءً على ملفك، إليك المسار المقترح من صندوق خليفة:"]
            for item in top:
                tag = {"now": "الآن", "next": "التالي", "later": "لاحقًا"}[item["bucket"]]
                lines.append(f"• [{tag}] {item['name_ar']} — {item['description_ar'][:140]}...")
        else:
            lines = ["Based on your profile, here's your recommended KFED pathway:"]
            for item in top:
                tag = item["bucket"].capitalize()
                lines.append(f"• [{tag}] {item['name_en']} — {item['description_en'][:140]}...")

        if escalation:
            lines.append(
                "\nتنبيه: تم أيضًا تحويل حالتك إلى مستشار بشري في صندوق خليفة للمتابعة."
                if lang == "ar" else
                "\nNote: this case has also been escalated to a human KFED advisor for follow-up."
            )
        return "\n".join(lines)

    def compose_welcome_message(self, profile, pathway, escalation):
        """Used once, right after the profiling quiz completes."""
        lang = profile.get("language", "en")
        name = profile.get("business_name") or ""
        if lang == "ar":
            greeting = f"شكرًا لك{(' عن ' + name) if name else ''}! بناءً على إجاباتك، إليك مسارك المقترح من صندوق خليفة:"
        else:
            greeting = f"Thanks{(' — ' + name) if name else ''}! Based on your answers, here's your recommended KFED pathway:"
        body = self.compose_message(profile, pathway, escalation)
        # strip the generic opening line from compose_message and use our warmer one
        body_lines = body.split("\n")[1:] if body else []
        return "\n".join([greeting] + body_lines)


# ---------------------------------------------------------------------------
# Agent 5: Progress Tracker (autonomous)
# ---------------------------------------------------------------------------

class ProgressTrackerAgent:
    name = "ProgressTrackerAgent"

    def persist(self, state, beneficiary_id, profile, pathway):
        state["profiles"][beneficiary_id] = profile
        state.setdefault("pathways", {}).setdefault(beneficiary_id, pathway)
        # merge: keep existing milestone_status if item already tracked
        existing = {p["program_id"]: p for p in state["pathways"][beneficiary_id]}
        merged = []
        seen = set()
        for item in pathway:
            if item["program_id"] in existing:
                kept = existing[item["program_id"]]
                kept.update({k: v for k, v in item.items() if k != "milestone_status"})
                merged.append(kept)
            else:
                merged.append(item)
            seen.add(item["program_id"])
        for pid, old in existing.items():
            if pid not in seen:
                merged.append(old)
        state["pathways"][beneficiary_id] = merged
        return state

    def mark_progress(self, state, beneficiary_id, program_id, status):
        items = state.get("pathways", {}).get(beneficiary_id, [])
        for item in items:
            if item["program_id"] == program_id:
                item["milestone_status"] = status
        return state

    def check_alerts(self, pathway):
        """Runs on its own — no user question required — scanning each
        recommended programme's real key_dates against today's date."""
        alerts = []
        for item in pathway:
            kd = item.get("key_dates")
            if not kd:
                continue
            close = kd.get("applications_close")
            open_ = kd.get("applications_open")
            if close:
                close_date = datetime.date.fromisoformat(close)
                days_left = (close_date - TODAY).days
                if 0 <= days_left <= 30:
                    alerts.append({
                        "program_id": item["program_id"], "type": "deadline_soon",
                        "en": f"Applications for {item['name_en']} close in {days_left} day(s) ({close}).",
                        "ar": f"تُغلق باب التقديم لبرنامج {item['name_ar']} خلال {days_left} يوم/أيام ({close}).",
                    })
                elif days_left < 0:
                    alerts.append({
                        "program_id": item["program_id"], "type": "edition_closed",
                        "en": f"{item['name_en']} applications are closed for this edition. "
                              f"{kd.get('next_edition_hint', 'Watch for the next edition.')}",
                        "ar": f"باب التقديم لبرنامج {item['name_ar']} مغلق لهذه النسخة.",
                    })
            start = kd.get("programme_start")
            if start:
                start_date = datetime.date.fromisoformat(start)
                days_to_start = (start_date - TODAY).days
                if 0 <= days_to_start <= 30:
                    alerts.append({
                        "program_id": item["program_id"], "type": "starting_soon",
                        "en": f"{item['name_en']} starts in {days_to_start} day(s) ({start}).",
                        "ar": f"يبدأ برنامج {item['name_ar']} خلال {days_to_start} يوم/أيام ({start}).",
                    })
        return alerts


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, kb_path=None):
        self.kb = KnowledgeBase(kb_path)
        self.profiler = ProfilerAgent()
        self.escalator = EscalationAgent()
        self.retriever = RetrieverAgent(self.kb)
        self.recommender = RecommenderAgent()
        self.tracker = ProgressTrackerAgent()

    # -- Profiling quiz -----------------------------------------------------

    def start_quiz(self, beneficiary_id, language):
        state = store.load_state()
        quiz_progress = state.setdefault("quiz_progress", {})
        progress = quiz_progress.get(beneficiary_id)

        if progress and progress.get("complete"):
            # Already profiled — resume straight into their existing pathway.
            return {
                "quiz_complete": True,
                "resumed": True,
                "reply": None,
                "profile": state["profiles"].get(beneficiary_id),
                "pathway": state["pathways"].get(beneficiary_id, []),
                "alerts": self.tracker.check_alerts(state["pathways"].get(beneficiary_id, [])),
            }

        if not progress:
            progress = {"index": 0, "answers": {}, "complete": False}
            quiz_progress[beneficiary_id] = progress
            store.save_state(state)

        question = QUIZ_QUESTIONS[progress["index"]]
        return {
            "quiz_complete": False,
            "step": progress["index"] + 1,
            "total_steps": len(QUIZ_QUESTIONS),
            "question": localize_question(question, language),
        }

    def submit_quiz_answer(self, beneficiary_id, question_id, answer, language):
        state = store.load_state()
        quiz_progress = state.setdefault("quiz_progress", {})
        progress = quiz_progress.setdefault(beneficiary_id, {"index": 0, "answers": {}, "complete": False})

        expected = QUIZ_QUESTIONS[progress["index"]]
        if expected["id"] != question_id:
            # Out-of-order submission (e.g. stale client) — ignore and just
            # resend the question the state machine is actually expecting.
            store.save_state(state)
            return {
                "quiz_complete": False,
                "step": progress["index"] + 1,
                "total_steps": len(QUIZ_QUESTIONS),
                "question": localize_question(expected, language),
            }

        progress["answers"][question_id] = answer
        progress["index"] += 1
        store.save_state(state)

        if progress["index"] < len(QUIZ_QUESTIONS):
            next_q = QUIZ_QUESTIONS[progress["index"]]
            store.save_state(state)
            return {
                "quiz_complete": False,
                "step": progress["index"] + 1,
                "total_steps": len(QUIZ_QUESTIONS),
                "question": localize_question(next_q, language),
            }

        # Quiz finished — run the full pipeline from the structured answers.
        profile = self.profiler.from_quiz(beneficiary_id, progress["answers"], language)
        query_text = f"{profile['business_name']} {profile['context']}"
        retrieved = self.retriever.run(profile, query_text)
        escalation_text = f"{profile['business_name']}. {profile['context']}"
        escalation = self.escalator.check(profile, escalation_text, retrieved)
        pathway = self.recommender.build(profile, retrieved)

        state = self.tracker.persist(state, beneficiary_id, profile, pathway)
        if escalation:
            state.setdefault("escalations", []).append(escalation)
        progress["complete"] = True
        store.save_state(state)

        alerts = self.tracker.check_alerts(pathway)
        reply_text = self.recommender.compose_welcome_message(profile, pathway, escalation)

        return {
            "quiz_complete": True,
            "resumed": False,
            "reply": reply_text,
            "language": profile.get("language"),
            "profile": profile,
            "pathway": state["pathways"][beneficiary_id],
            "retrieved": [{"id": d["id"], "name_en": d["name_en"], "score": s} for d, s in retrieved],
            "escalation": escalation,
            "alerts": alerts,
            "llm_live": llm.is_live(),
        }

    def restart_quiz(self, beneficiary_id):
        state = store.load_state()
        state.setdefault("quiz_progress", {})[beneficiary_id] = {"index": 0, "answers": {}, "complete": False}
        store.save_state(state)
        return {"ok": True}

    # -- Free-text follow-up chat (post-profiling) ---------------------------

    def handle_turn(self, beneficiary_id, message):
        state = store.load_state()
        existing_profile = state["profiles"].get(beneficiary_id)

        profile = self.profiler.run(beneficiary_id, message, existing_profile)
        retrieved = self.retriever.run(profile, message)
        escalation = self.escalator.check(profile, message, retrieved)
        pathway = self.recommender.build(profile, retrieved)

        state = self.tracker.persist(state, beneficiary_id, profile, pathway)
        if escalation:
            state.setdefault("escalations", []).append(escalation)
        store.save_state(state)

        alerts = self.tracker.check_alerts(pathway)
        reply_text = self.recommender.compose_message(profile, pathway, escalation)

        return {
            "reply": reply_text,
            "language": profile.get("language"),
            "profile": profile,
            "pathway": state["pathways"][beneficiary_id],
            "retrieved": [{"id": d["id"], "name_en": d["name_en"], "score": s} for d, s in retrieved],
            "escalation": escalation,
            "alerts": alerts,
            "llm_live": llm.is_live(),
        }

    def mark_progress(self, beneficiary_id, program_id, status):
        state = store.load_state()
        state = self.tracker.mark_progress(state, beneficiary_id, program_id, status)
        store.save_state(state)
        return state["pathways"].get(beneficiary_id, [])

    def get_beneficiary(self, beneficiary_id):
        state = store.load_state()
        return {
            "profile": state["profiles"].get(beneficiary_id),
            "pathway": state["pathways"].get(beneficiary_id, []),
        }
