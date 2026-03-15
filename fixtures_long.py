"""Long synthetic conversation: 200 messages, 8 topics, 40 planted facts.

Designed to create compression pressure. At 190 cold messages,
a single flat summary must drop details. Union-find's per-cluster
summaries each cover ~25 messages — less compression per cluster.

v3: timestamps for each message (2026-03-14 workday, ~2.4 min apart).
"""

from datetime import datetime, timedelta, timezone

# 8 topics, 5 facts each = 40 facts.

FACTS_LONG = [
    # Topic 1: Database migration
    {"topic": "db", "msg_index": 2, "question": "What is the source database engine being migrated from?", "answer": "MySQL 5.7.44"},
    {"topic": "db", "msg_index": 8, "question": "What is the target PostgreSQL version for the migration?", "answer": "16.3"},
    {"topic": "db", "msg_index": 18, "question": "How many tables need to be migrated?", "answer": "347"},
    {"topic": "db", "msg_index": 45, "question": "What tool is being used for the schema conversion?", "answer": "pgloader 3.6.10"},
    {"topic": "db", "msg_index": 92, "question": "What is the estimated downtime window for the migration?", "answer": "4 hours starting at 02:00 UTC Saturday"},
    # Topic 2: Authentication rewrite
    {"topic": "auth", "msg_index": 5, "question": "What is the new JWT signing algorithm?", "answer": "EdDSA with Ed25519"},
    {"topic": "auth", "msg_index": 14, "question": "What is the token expiry time?", "answer": "15 minutes for access, 7 days for refresh"},
    {"topic": "auth", "msg_index": 35, "question": "What Redis key prefix is used for revoked tokens?", "answer": "auth:revoked:"},
    {"topic": "auth", "msg_index": 68, "question": "What is the maximum number of concurrent sessions per user?", "answer": "5"},
    {"topic": "auth", "msg_index": 110, "question": "What rate limit applies to the /auth/token endpoint?", "answer": "10 requests per minute per IP"},
    # Topic 3: Search infrastructure
    {"topic": "search", "msg_index": 10, "question": "What search engine is being deployed?", "answer": "Meilisearch 1.11"},
    {"topic": "search", "msg_index": 23, "question": "How many documents are in the initial index?", "answer": "2.3 million"},
    {"topic": "search", "msg_index": 52, "question": "What is the p99 search latency target?", "answer": "50ms"},
    {"topic": "search", "msg_index": 85, "question": "What field is used as the primary ranking criterion?", "answer": "freshness_score descending"},
    {"topic": "search", "msg_index": 130, "question": "What is the maximum filterable attribute count?", "answer": "64 per index"},
    # Topic 4: CI/CD pipeline
    {"topic": "ci", "msg_index": 12, "question": "What CI runner image is being used?", "answer": "ubuntu-24.04-arm64"},
    {"topic": "ci", "msg_index": 28, "question": "What is the test parallelism level?", "answer": "8 shards"},
    {"topic": "ci", "msg_index": 55, "question": "What is the artifact retention period?", "answer": "90 days"},
    {"topic": "ci", "msg_index": 95, "question": "What is the deployment approval requirement for production?", "answer": "2 approvals from the platform-oncall group"},
    {"topic": "ci", "msg_index": 145, "question": "What is the maximum build time before timeout?", "answer": "45 minutes"},
    # Topic 5: Billing system
    {"topic": "billing", "msg_index": 15, "question": "What payment processor is being integrated?", "answer": "Stripe with Connect for marketplace payouts"},
    {"topic": "billing", "msg_index": 32, "question": "What is the platform fee percentage?", "answer": "2.9% plus 30 cents per transaction"},
    {"topic": "billing", "msg_index": 60, "question": "What webhook endpoint receives payment events?", "answer": "/api/v3/webhooks/stripe"},
    {"topic": "billing", "msg_index": 100, "question": "What is the invoice generation schedule?", "answer": "1st and 15th of each month at 06:00 UTC"},
    {"topic": "billing", "msg_index": 155, "question": "What is the dunning retry schedule for failed payments?", "answer": "retry at 1, 3, 5, and 7 days then suspend"},
    # Topic 6: Monitoring
    {"topic": "monitor", "msg_index": 20, "question": "What metrics backend is being used?", "answer": "VictoriaMetrics 1.106"},
    {"topic": "monitor", "msg_index": 38, "question": "What is the scrape interval for application metrics?", "answer": "15 seconds"},
    {"topic": "monitor", "msg_index": 70, "question": "What PagerDuty escalation policy is used for P1 alerts?", "answer": "platform-critical with 5-minute escalation"},
    {"topic": "monitor", "msg_index": 105, "question": "What is the error rate threshold that triggers an alert?", "answer": "5% of requests over a 5-minute window"},
    {"topic": "monitor", "msg_index": 160, "question": "What Grafana dashboard ID shows the billing pipeline health?", "answer": "dashboard-4471"},
    # Topic 7: Mobile app
    {"topic": "mobile", "msg_index": 25, "question": "What minimum iOS version does the app support?", "answer": "iOS 16.0"},
    {"topic": "mobile", "msg_index": 42, "question": "What state management library is used in the iOS app?", "answer": "TCA (The Composable Architecture) 1.17"},
    {"topic": "mobile", "msg_index": 75, "question": "What is the offline sync conflict resolution strategy?", "answer": "last-write-wins with vector clocks"},
    {"topic": "mobile", "msg_index": 115, "question": "What deep link scheme does the app register?", "answer": "myapp://"},
    {"topic": "mobile", "msg_index": 170, "question": "What is the app binary size budget?", "answer": "35MB after thinning"},
    # Topic 8: Data pipeline
    {"topic": "data", "msg_index": 30, "question": "What streaming platform is used for event ingestion?", "answer": "Redpanda 24.3"},
    {"topic": "data", "msg_index": 48, "question": "What is the Kafka topic partition count for user-events?", "answer": "32"},
    {"topic": "data", "msg_index": 80, "question": "What serialization format is used for events?", "answer": "Avro with schema registry"},
    {"topic": "data", "msg_index": 120, "question": "What is the data retention period in the lake?", "answer": "7 years for financial, 2 years for behavioral"},
    {"topic": "data", "msg_index": 180, "question": "What is the batch job schedule for the daily aggregation?", "answer": "cron 0 4 * * * UTC"},
]

assert len(FACTS_LONG) == 40


def _generate_conversation() -> tuple[list[str], list[str]]:
    """Build a 200-message conversation with facts planted at specified indices.

    Returns (messages, timestamps) — parallel lists.
    Timestamps span a workday: 2026-03-14 09:00–17:00 UTC, ~2.4 min apart.
    """

    # Fact content by index for easy lookup
    fact_at: dict[int, str] = {}

    # Topic 1: Database migration
    fact_at[2] = "We're migrating from MySQL 5.7.44 to Postgres. The source DB has been running since 2019."
    fact_at[8] = "Target is PostgreSQL 16.3. We need the new JSONB improvements for the config tables."
    fact_at[18] = "Just finished the audit — 347 tables need to be migrated. About 40 have custom triggers."
    fact_at[45] = "I've been testing pgloader 3.6.10 for the schema conversion. It handles most MySQL-to-Postgres type mapping automatically."
    fact_at[92] = "Migration window is 4 hours starting at 02:00 UTC Saturday. We'll drain connections at 01:45."

    # Topic 2: Authentication
    fact_at[5] = "New auth system uses EdDSA with Ed25519 for JWT signing. Much faster than RS256."
    fact_at[14] = "Token lifetimes: 15 minutes for access, 7 days for refresh. Refresh tokens are rotated on use."
    fact_at[35] = "Revoked tokens go to Redis under the auth:revoked: key prefix. TTL matches the token's remaining lifetime."
    fact_at[68] = "We're capping concurrent sessions at 5 per user. Oldest session gets killed when a 6th login happens."
    fact_at[110] = "Rate limit on /auth/token is 10 requests per minute per IP. Prevents credential stuffing."

    # Topic 3: Search
    fact_at[10] = "We're deploying Meilisearch 1.11 for the product search. It handles typo tolerance out of the box."
    fact_at[23] = "Initial index load is 2.3 million documents. Most are product listings from the last 3 years."
    fact_at[52] = "Search latency target is p99 under 50ms. Current prototype hits 35ms on the test dataset."
    fact_at[85] = "Primary ranking criterion is freshness_score descending. Newer products rank higher by default."
    fact_at[130] = "Meilisearch caps filterable attributes at 64 per index. We're using 41 right now."

    # Topic 4: CI/CD
    fact_at[12] = "CI runners are on ubuntu-24.04-arm64 images. ARM gives us 40% better price-performance."
    fact_at[28] = "Test suite runs in 8 shards. Each shard gets its own DB and Redis instance."
    fact_at[55] = "Artifact retention is 90 days. After that, only the build manifest is kept."
    fact_at[95] = "Production deploys require 2 approvals from the platform-oncall group. No self-approvals."
    fact_at[145] = "Build timeout is 45 minutes. If it takes longer, something is wrong and we'd rather fail fast."

    # Topic 5: Billing
    fact_at[15] = "Billing integration is Stripe with Connect for marketplace payouts. Sellers get direct deposits."
    fact_at[32] = "Platform fee is 2.9% plus 30 cents per transaction. Standard Stripe pricing, we pass it through."
    fact_at[60] = "Stripe webhooks hit /api/v3/webhooks/stripe. We verify the signature with the endpoint secret."
    fact_at[100] = "Invoices generate on the 1st and 15th of each month at 06:00 UTC. PDF delivery via email."
    fact_at[155] = "Dunning for failed payments: retry at 1, 3, 5, and 7 days then suspend the account."

    # Topic 6: Monitoring
    fact_at[20] = "Metrics backend is VictoriaMetrics 1.106. Drop-in Prometheus replacement with better compression."
    fact_at[38] = "Application metrics scrape interval is 15 seconds. Infrastructure metrics every 60."
    fact_at[70] = "P1 alerts use the platform-critical PagerDuty escalation policy with 5-minute escalation."
    fact_at[105] = "Error rate alert fires when 5% of requests fail over a 5-minute window."
    fact_at[160] = "Billing pipeline health is on Grafana dashboard-4471. It tracks invoice generation and webhook delivery."

    # Topic 7: Mobile
    fact_at[25] = "iOS app minimum is iOS 16.0. We dropped 15 support last quarter — only 3% of users."
    fact_at[42] = "State management is TCA (The Composable Architecture) 1.17. Gives us deterministic testing."
    fact_at[75] = "Offline sync uses last-write-wins with vector clocks. We tried CRDTs but the merge logic was too complex."
    fact_at[115] = "Deep links use the myapp:// scheme. Universal links are configured for the main domain too."
    fact_at[170] = "App binary size budget is 35MB after thinning. We're at 31MB right now."

    # Topic 8: Data pipeline
    fact_at[30] = "Event ingestion runs on Redpanda 24.3. Kafka-compatible but uses Raft instead of ZooKeeper."
    fact_at[48] = "user-events topic has 32 partitions. Keyed by user_id for ordering guarantees."
    fact_at[80] = "Events are serialized as Avro with schema registry. No more JSON parsing surprises."
    fact_at[120] = "Data retention: 7 years for financial, 2 years for behavioral. Compliance requirement."
    fact_at[180] = "Daily aggregation batch job runs at cron 0 4 * * * UTC. Takes about 20 minutes."

    # Filler messages — realistic conversation connective tissue
    fillers = [
        "Got it, makes sense.",
        "I'll update the PR with that.",
        "Let me check the docs real quick.",
        "That matches what I saw in the staging logs.",
        "We should document this somewhere.",
        "I'll add that to the runbook.",
        "Good call. What else?",
        "Let me pull up the dashboard.",
        "I tested this locally and it works.",
        "We need to loop in the platform team on this.",
        "I'll file a ticket for the follow-up.",
        "That's consistent with what we decided last week.",
        "Let me verify that number.",
        "Confirmed. Moving on.",
        "I'll add a test for that edge case.",
        "That's a good point — I hadn't considered that.",
        "Let me check if that's already in the backlog.",
        "I'll send the PR after this call.",
        "Do we have a runbook for that scenario?",
        "I think we covered that in the design doc.",
        "Let me double-check the config.",
        "That should be in the environment variables.",
        "I'll update the README.",
        "We should probably automate that.",
        "I'll write a migration script for it.",
        "Let me check the error logs.",
        "That's expected behavior, actually.",
        "I'll add it to the sprint board.",
        "Let me pull that up.",
        "We discussed this in the architecture review.",
        "I'll run the load test after the fix is in.",
        "That aligns with the SLA.",
        "Let me sync with the security team.",
        "I'll update the Terraform config.",
        "We should add a canary for that.",
        "Let me check the metrics.",
        "That's within the error budget.",
        "I'll add monitoring for it.",
        "Let me look at the trace.",
        "We need to update the client SDK too.",
    ]

    conversation = []
    timestamps = []
    filler_idx = 0
    start = datetime(2026, 3, 14, 9, 0, 0, tzinfo=timezone.utc)
    interval = timedelta(seconds=480 / 200 * 60)  # ~2.4 min apart over 8 hours

    for i in range(200):
        ts = start + interval * i
        timestamps.append(ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
        if i in fact_at:
            conversation.append(fact_at[i])
        else:
            conversation.append(fillers[filler_idx % len(fillers)])
            filler_idx += 1

    return conversation, timestamps


CONVERSATION_LONG, TIMESTAMPS_LONG = _generate_conversation()
assert len(CONVERSATION_LONG) == 200
assert len(TIMESTAMPS_LONG) == 200

# Verify all fact indices are valid
for fact in FACTS_LONG:
    assert 0 <= fact["msg_index"] < len(CONVERSATION_LONG), f"Bad index: {fact['msg_index']}"
