# fasta2a_server.py DOX

## Purpose

Serve Agent Zero through the authenticated dynamic A2A proxy. Reuse a conversation across tasks while keeping unrelated contexts separate.

## Contracts

- `AgentZeroStorage` extends FastA2A's process-local task/context store. Its conversation entries contain an Agent Zero context ID, project, execution lock, pending count and activity timestamp. It never aliases caller-supplied A2A IDs to existing user chats.
- A message without `contextId` starts a fresh conversation. A supplied ID must exist and match the project's scope before a new task is accepted. Unavailable IDs produce JSON-RPC invalid-parameters errors; there is no silent fallback.
- `AgentZeroWorker` reuses one BACKGROUND context per conversation and serializes its turns. Different conversations are independent. The first agent configuration/profile remains in effect. Failed/cancelled tasks do not delete the conversation.
- Cancellation stops only the matching active task or skips matching queued work. Late worker results cannot overwrite terminal task states.
- Conversations expire after 24 idle hours. The existing API-chat cleanup uses `lifetime_hours`; lazy pruning also removes stale mappings and resets expired contexts. Pending/active work is not pruned. Shutdown/reconfiguration clears retained contexts.
- Conversations remain BACKGROUND and are not written to chat persistence. Process restart/reconfiguration invalidates conversation IDs; this does not provide crash recovery of in-flight computation.
- Task timestamps are UTC RFC 3339. Completed replies are text artifacts and history messages. Reading a shortened history does not mutate stored history.
- `AgentZeroFastA2A` serves both agent-card paths. Request-specific RPC URLs preserve the auth/project prefix and are not cached across projects. The existing streaming-capability wrapper remains unchanged.
- `DynamicA2AProxy` retains the existing token/header authorization. Project injection buffers a complete ASGI request before returning rewritten bytes, including chunked bodies. Never log token-bearing URLs.
- The transport does not alter provider settings or agent source code; retained conversations are BACKGROUND runtime state.

## Verification

Run `python -m pytest tests/test_fasta2a_server.py tests/test_a2a_conversations.py tests/test_framework_dependencies.py -q` in the framework environment with development requirements. Tests use fake agents, real FastA2A and the official SDK over an in-process HTTP transport; no keys, external services or local user state are required.
