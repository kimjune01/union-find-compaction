"""Synthetic conversation with planted facts and recall questions.

Written before experimental code. The conversation simulates a coding
session with 5 interleaved topics. Each topic has 4 planted facts
that are specific enough to be lost by vague summarization.
"""

# Ground truth: 20 facts, 4 per topic.
# Each fact has: topic, message_index, fact, question, answer.

FACTS = [
    # Topic 1: Project setup
    {
        "topic": "setup",
        "msg_index": 1,
        "question": "What version of PostgreSQL is the project using?",
        "answer": "16.2",
    },
    {
        "topic": "setup",
        "msg_index": 3,
        "question": "What port is the database running on?",
        "answer": "5433",
    },
    {
        "topic": "setup",
        "msg_index": 12,
        "question": "What is the name of the database schema?",
        "answer": "tenant_core",
    },
    {
        "topic": "setup",
        "msg_index": 22,
        "question": "What ORM version is being used?",
        "answer": "SQLAlchemy 2.0.35",
    },
    # Topic 2: Bug hunt
    {
        "topic": "bug",
        "msg_index": 5,
        "question": "Which file contained the async bug?",
        "answer": "auth_handler.py",
    },
    {
        "topic": "bug",
        "msg_index": 7,
        "question": "What line number was the missing await on?",
        "answer": "247",
    },
    {
        "topic": "bug",
        "msg_index": 15,
        "question": "What exception type was being raised by the async bug?",
        "answer": "RuntimeError: coroutine was never awaited",
    },
    {
        "topic": "bug",
        "msg_index": 25,
        "question": "What was the root cause of the auth bug?",
        "answer": "verify_token was called without await so it returned a coroutine object instead of a bool",
    },
    # Topic 3: API design
    {
        "topic": "api",
        "msg_index": 8,
        "question": "What is the API rate limit per key?",
        "answer": "150 requests per minute",
    },
    {
        "topic": "api",
        "msg_index": 11,
        "question": "What authentication scheme does the API use?",
        "answer": "HMAC-SHA256 with rotating keys",
    },
    {
        "topic": "api",
        "msg_index": 19,
        "question": "What is the path for the batch inference endpoint?",
        "answer": "/v2/batch/infer",
    },
    {
        "topic": "api",
        "msg_index": 30,
        "question": "What format does the batch endpoint return results in?",
        "answer": "NDJSON streaming with a final summary object",
    },
    # Topic 4: Deployment
    {
        "topic": "deploy",
        "msg_index": 9,
        "question": "What is the staging server IP address?",
        "answer": "10.0.3.47",
    },
    {
        "topic": "deploy",
        "msg_index": 14,
        "question": "What SSH port does the staging server use?",
        "answer": "2222",
    },
    {
        "topic": "deploy",
        "msg_index": 24,
        "question": "What is the path to the deploy script?",
        "answer": "/opt/deploy/run_migration.sh",
    },
    {
        "topic": "deploy",
        "msg_index": 33,
        "question": "What is the rollback command?",
        "answer": "kubectl rollout undo deployment/api-server --to-revision=3",
    },
    # Topic 5: Code review
    {
        "topic": "review",
        "msg_index": 10,
        "question": "Which function has the race condition?",
        "answer": "merge_accounts",
    },
    {
        "topic": "review",
        "msg_index": 17,
        "question": "What balance threshold triggers the race condition?",
        "answer": "10000",
    },
    {
        "topic": "review",
        "msg_index": 28,
        "question": "What type of lock was proposed to fix the race condition?",
        "answer": "SELECT FOR UPDATE SKIP LOCKED",
    },
    {
        "topic": "review",
        "msg_index": 35,
        "question": "What table does the merge_accounts function write to?",
        "answer": "ledger_entries",
    },
]

# The synthetic conversation. 50 messages, interleaved topics.
# Messages at FACTS[i]["msg_index"] contain the planted fact.

CONVERSATION = [
    # 0
    "Hey, I'm setting up the new project. Let me share the infrastructure details.",
    # 1 — setup: PostgreSQL 16.2
    "We're running PostgreSQL 16.2 for the main datastore. The team upgraded from 15 last sprint.",
    # 2
    "Got it. I'll start with the connection config.",
    # 3 — setup: port 5433
    "Make sure you use port 5433, not the default. We moved it after the security audit.",
    # 4
    "Noted. What about the application layer?",
    # 5 — bug: auth_handler.py
    "Before that — there's a bug I've been chasing in auth_handler.py. Authentication is failing intermittently.",
    # 6
    "Intermittent auth failures? That sounds like a concurrency issue.",
    # 7 — bug: line 247
    "Yeah, I narrowed it down to line 247. There's a missing await on the token verification call.",
    # 8 — api: rate limit 150/min
    "While you fix that — for the API design doc, we settled on 150 requests per minute per API key as the rate limit.",
    # 9 — deploy: staging IP
    "Also, the staging server is at 10.0.3.47. I'll send you the SSH config.",
    # 10 — review: merge_accounts race condition
    "I found something in the code review. The merge_accounts function has a race condition when called concurrently.",
    # 11 — api: HMAC-SHA256
    "For the API auth scheme, we're going with HMAC-SHA256 with rotating keys. No bearer tokens.",
    # 12 — setup: tenant_core schema
    "Back to the DB setup — the schema name is tenant_core. All application tables go there.",
    # 13
    "Makes sense. I'll update the migration scripts.",
    # 14 — deploy: SSH port 2222
    "For the staging server SSH, use port 2222. The bastion host proxies through there.",
    # 15 — bug: RuntimeError
    "The auth bug is throwing RuntimeError: coroutine was never awaited. Classic async mistake.",
    # 16
    "That confirms it. The verify function is async but being called synchronously somewhere.",
    # 17 — review: threshold 10000
    "On the race condition — it only triggers when the account balance exceeds 10000. Below that, the fast path avoids the lock.",
    # 18
    "So it's a conditional race. That's harder to reproduce in tests.",
    # 19 — api: /v2/batch/infer
    "The batch inference endpoint path is /v2/batch/infer. It handles up to 1000 items per request.",
    # 20
    "Good. I'll add that to the OpenAPI spec.",
    # 21
    "Let me also document the error codes while we're at it.",
    # 22 — setup: SQLAlchemy 2.0.35
    "One more setup detail — we're pinned to SQLAlchemy 2.0.35. Don't upgrade yet, 2.1 has a breaking change in the session API.",
    # 23
    "Understood. I'll lock it in pyproject.toml.",
    # 24 — deploy: /opt/deploy/run_migration.sh
    "The deploy script is at /opt/deploy/run_migration.sh. It handles both schema migration and data backfill.",
    # 25 — bug: root cause
    "Root cause confirmed on the auth bug: verify_token was called without await so it returned a coroutine object instead of a bool. The truthy check on the coroutine always passed.",
    # 26
    "That explains why it was intermittent — it only failed when the token was actually invalid.",
    # 27
    "Right. The coroutine object is truthy, so valid tokens worked by accident.",
    # 28 — review: SELECT FOR UPDATE SKIP LOCKED
    "For the merge_accounts race condition, I'm proposing SELECT FOR UPDATE SKIP LOCKED. It lets concurrent calls proceed on different rows without blocking.",
    # 29
    "That's clean. Much better than a global advisory lock.",
    # 30 — api: NDJSON streaming
    "The batch endpoint returns results as NDJSON streaming with a final summary object. Each line is one inference result.",
    # 31
    "Streaming NDJSON — that means the client can process results as they arrive.",
    # 32
    "Exactly. No need to buffer the entire response.",
    # 33 — deploy: rollback command
    "If the deploy goes wrong, rollback with: kubectl rollout undo deployment/api-server --to-revision=3",
    # 34
    "Revision 3 specifically? Or the previous revision?",
    # 35 — review: ledger_entries table
    "The merge_accounts function writes to the ledger_entries table. That's where the double-write happens during the race.",
    # 36
    "So the fix needs to lock ledger_entries rows, not the accounts table.",
    # 37
    "Correct. The accounts table read is safe. The write to ledger_entries is where contention happens.",
    # 38
    "I'll write a test that hammers merge_accounts with concurrent requests above the 10k threshold.",
    # 39
    "Good idea. Use pytest-asyncio with 50 concurrent tasks.",
    # 40
    "Should we add a circuit breaker on the batch endpoint too?",
    # 41
    "Not yet. Let's get the rate limiter right first. 150/min should be enough for the beta.",
    # 42
    "Fair. I'll focus on the auth fix and the merge_accounts lock.",
    # 43
    "The auth fix is one line — add await on line 247. The merge fix is more involved.",
    # 44
    "I'll PR the auth fix first since it's blocking QA.",
    # 45
    "Do that. I'll review the merge_accounts PR when it's ready.",
    # 46
    "Should I update the staging deployment after the auth fix?",
    # 47
    "Yes, SSH into 10.0.3.47 on port 2222 and run the deploy script.",
    # 48
    "Will do. Anything else before I start?",
    # 49
    "That's everything. Let's sync again after the PRs are up.",
]

assert len(CONVERSATION) == 50
assert len(FACTS) == 20

# Verify all fact indices are valid
for fact in FACTS:
    assert 0 <= fact["msg_index"] < len(CONVERSATION), f"Bad index: {fact['msg_index']}"
