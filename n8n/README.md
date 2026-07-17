# n8n workflow (alternative / bonus orchestration layer)

The **coded prototype in `backend/`** is what actually runs the multi-agent
pipeline (Profiler → Escalation → RAG Retriever → Recommender → Progress
Tracker) and is what you should use for the Day 5 live demo — it needs zero
installs and zero external accounts.

This `n8n` folder is a **bonus artifact** for teams that want to show judges
they explored a no-code/low-code orchestration layer too, per the challenge's
suggested tool list (n8n, Lovable, etc.). It does **not** reimplement the
agent logic in n8n — instead it treats the Python backend as "the brain" and
n8n as the front door / integration layer, which is a realistic pattern for
how KFED would actually wire this into WhatsApp, a call-centre tool, or the
Qudorat platform later.

## What's in `kfed-advisor-workflow.json`

```
Webhook (POST /kfed-advisor)
   -> HTTP Request -> POST http://localhost:8420/api/chat   (runs all 5 agents)
   -> IF escalation present?
        yes -> Set (compose human-advisor handoff summary) -> Respond (escalated)
        no  -> Respond (normal, with pathway + alerts)
```

## Import steps

1. Start the Python backend first: `python3 -m backend.app` (from the project root).
2. In n8n: **Workflows → Import from File** → select `kfed-advisor-workflow.json`.
3. Activate the workflow. It exposes `POST /webhook/kfed-advisor` on your n8n instance.
4. Test: `curl -X POST http://<your-n8n-host>/webhook/kfed-advisor -H "Content-Type: application/json" -d '{"beneficiary_id":"B-1042","message":"أحتاج مساعدة في التصدير"}'`

## Swap points for a real deployment

- **HTTP Request node URL**: point it at your hosted backend instead of `localhost:8420`.
- **"Compose Human Advisor Handoff Summary" node**: currently a `Set` node so the
  workflow imports without requiring credentials. Replace its output connection
  with a real Slack, Microsoft Teams, or Email node (KFED advisor channel) to
  actually page a human — the composed summary text is already built for you.
- **Webhook auth**: add n8n's built-in webhook authentication (header/basic auth)
  before exposing this publicly.
