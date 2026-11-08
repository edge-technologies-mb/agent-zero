"""Real FastA2A/SDK transport with a deterministic Agent Zero substitute."""
import asyncio
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import types
import uuid

import fasta2a  # Load real transport before legacy tests stub Starlette.
import httpx
import pytest
from a2a.client.client import ClientConfig
from a2a.client.client_factory import create_client
from a2a.client.card_resolver import parse_agent_card
from a2a.helpers.proto_helpers import new_text_part
from a2a.types import a2a_pb2 as pb
from fasta2a.broker import InMemoryBroker


@pytest.fixture
def server(monkeypatch):
    class Context:
        contexts = {}
        started = None
        release = None

        def __init__(self, config, **kwargs):
            self.id = uuid.uuid4().hex
            self.history = []
            self.data = {}
            self.log = types.SimpleNamespace(log=lambda **kwargs: None)
            self.contexts[self.id] = self
            self.killed = False

        @classmethod
        def get(cls, id): return cls.contexts.get(id)
        @classmethod
        def remove(cls, id): cls.contexts.pop(id, None)
        def set_data(self, key, value): self.data[key] = value
        def reset(self): self.kill_process()
        def kill_process(self):
            self.killed = True
            if self.release: self.release.set()

        def communicate(self, message):
            async def result():
                self.history.append(message.message)
                if message.message == "slow":
                    self.started.set()
                    await self.release.wait()
                if self.killed or message.message == "fail":
                    self.killed = False
                    raise RuntimeError("private failure")
                return " / ".join(self.history)
            return types.SimpleNamespace(result=result)

    def module(name, **values):
        value = types.ModuleType(name)
        value.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("agent", AgentContext=Context, UserMessage=lambda **kw: types.SimpleNamespace(**kw),
           AgentContextType=types.SimpleNamespace(BACKGROUND="background"))
    module("initialize", initialize_agent=lambda: {})
    module("helpers.settings", get_settings=lambda: {"a2a_server_enabled": True, "mcp_server_token": "test-token"})
    module("helpers.projects", activate_project=lambda *a: None)
    module("helpers.persist_chat", remove_chat=lambda *a: None)
    module("helpers.print_style", PrintStyle=lambda **kw: types.SimpleNamespace(print=lambda *a: None))
    name = "helpers.fasta2a_server"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "helpers/fasta2a_server.py")
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, adapter)
    spec.loader.exec_module(adapter)
    storage = adapter.AgentZeroStorage()
    broker = InMemoryBroker()
    worker = adapter.AgentZeroWorker(broker, storage)
    return types.SimpleNamespace(adapter=adapter, storage=storage, broker=broker, worker=worker, Context=Context)


async def submit(server, text, context_id=None, project=None):
    message = {"kind": "message", "message_id": uuid.uuid4().hex, "role": "user",
               "parts": [{"kind": "text", "text": text}], "metadata": {"project": project}}
    id = context_id or str(uuid.uuid4())
    if context_id is not None: message["context_id"] = context_id
    task = await server.storage.submit_task(id, message)
    return task, {"id": task["id"], "context_id": id, "message": message}


@pytest.mark.asyncio
async def test_reuse_isolated_fresh_contexts_and_artifacts(server):
    first, params = await submit(server, "nonce", project="sales")
    await server.worker.run_task(params)
    second, params = await submit(server, "recall", first["context_id"], "sales")
    await server.worker.run_task(params)
    assert second["id"] != first["id"]
    assert second["artifacts"][0]["parts"][0]["text"] == "nonce / recall"
    assert first["status"]["state"] == "completed"
    assert datetime.fromisoformat(second["status"]["timestamp"]).utcoffset() == timedelta(0)
    fresh, params = await submit(server, "recall", project="sales")
    await server.worker.run_task(params)
    assert fresh["context_id"] != first["context_id"]
    assert fresh["artifacts"][0]["parts"][0]["text"] == "recall"
    assert (await server.storage.load_task(second["id"], 0))["history"] == []
    assert len((await server.storage.load_task(second["id"]))["history"]) == 2


@pytest.mark.asyncio
async def test_invalid_expired_scope_and_restart_do_not_queue_work(server):
    first, params = await submit(server, "one", project="sales")
    await server.worker.run_task(params)
    for id, project in [("unknown", "sales"), (first["context_id"], "other")]:
        with pytest.raises(server.adapter.ConversationUnavailable): await submit(server, "bad", id, project)
    assert len(server.storage.tasks) == 1
    entry = await server.storage.load_context(first["context_id"])
    entry.last_used -= timedelta(hours=25)
    with pytest.raises(server.adapter.ConversationUnavailable): await submit(server, "expired", first["context_id"], "sales")
    assert not server.Context.contexts
    with pytest.raises(server.adapter.ConversationUnavailable):
        await server.adapter.AgentZeroStorage().submit_task(first["context_id"], params["message"])


@pytest.mark.asyncio
async def test_same_context_serializes_different_context_runs_and_cancel_is_scoped(server):
    server.Context.started = asyncio.Event()
    server.Context.release = asyncio.Event()
    first, p1 = await submit(server, "slow")
    running = asyncio.create_task(server.worker.run_task(p1))
    await server.Context.started.wait()
    entry = await server.storage.load_context(first["context_id"])
    entry.last_used -= timedelta(hours=25)
    queued, p2 = await submit(server, "queued", first["context_id"])
    waiting = asyncio.create_task(server.worker.run_task(p2))
    await asyncio.sleep(0)
    assert queued["status"]["state"] == "submitted"
    other, p3 = await submit(server, "independent")
    await server.worker.run_task(p3)
    assert other["status"]["state"] == "completed"
    await server.worker.cancel_task({"id": queued["id"]})
    assert first["status"]["state"] == "working"
    server.Context.release.set()
    await asyncio.gather(running, waiting)
    assert first["status"]["state"] == "completed"
    assert queued["status"]["state"] == "canceled"
    assert server.Context.get(entry.agent_context_id).history == ["slow"]
    follow, p4 = await submit(server, "follow", first["context_id"])
    await server.worker.run_task(p4)
    assert follow["artifacts"][0]["parts"][0]["text"] == "slow / follow"


@pytest.mark.asyncio
async def test_active_cancel_failure_and_subsequent_turn_retain_context(server):
    server.Context.started = asyncio.Event()
    server.Context.release = asyncio.Event()
    first, params = await submit(server, "slow")
    running = asyncio.create_task(server.worker.run_task(params))
    await server.Context.started.wait()
    await server.worker.cancel_task({"id": first["id"]})
    await running
    assert first["status"]["state"] == "canceled"
    failure, params = await submit(server, "fail", first["context_id"])
    await server.worker.run_task(params)
    assert failure["status"]["state"] == "failed"
    follow, params = await submit(server, "follow", first["context_id"])
    await server.worker.run_task(params)
    assert follow["status"]["state"] == "completed"
    server.storage.close_contexts()
    assert not server.Context.contexts
    assert not server.storage.contexts


@pytest.mark.asyncio
async def test_proxy_and_stock_sdk_two_turns_no_compatibility_shim(server):
    proxy = server.adapter.DynamicA2AProxy()
    transport = httpx.ASGITransport(app=proxy)
    base = "http://test/a2a/t-test-token/p-sales"
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            for suffix in ["agent.json", "agent-card.json"]:
                r = await http.get(base + "/.well-known/" + suffix)
                assert r.status_code == 200
                card_data = r.json()
                assert card_data["url"] == base + "/"
                assert card_data["protocolVersion"] == "0.3.0"
            # A second project's card must not inherit the first project's URL.
            other = await http.get("/a2a/t-test-token/p-other/.well-known/agent-card.json")
            assert other.json()["url"].endswith("/p-other/")
            assert (await http.get("/a2a/t-wrong/.well-known/agent-card.json")).status_code == 401
            client = await create_client(parse_agent_card(card_data), client_config=ClientConfig(streaming=False, httpx_client=http, accepted_output_modes=["text/plain"]))
            async def turn(text, context=""):
                request = pb.SendMessageRequest(message=pb.Message(message_id=uuid.uuid4().hex, role=pb.Role.ROLE_USER, parts=[new_text_part(text)], context_id=context))
                async for response in client.send_message(request):
                    task = response.task
                for _ in range(100):
                    if task.status.state == pb.TaskState.TASK_STATE_COMPLETED: return task
                    task = await client.get_task(pb.GetTaskRequest(id=task.id))
                    await asyncio.sleep(0)
                pytest.fail("Task did not complete")
            first = await turn("nonce")
            second = await turn("recall", first.context_id)
            assert first.id != second.id
            assert first.context_id == second.context_id
            assert second.artifacts[0].parts[0].text == "nonce / recall"
            bad = await http.post(base + "/", json={"jsonrpc":"2.0","id":"invalid","method":"message/send","params":{"message":{"kind":"message","messageId":"bad","contextId":"unknown","role":"user","parts":[{"kind":"text","text":"bad"}]}}})
            assert bad.json()["error"]["code"] == -32602
            mismatch = await http.post("/a2a/t-test-token/p-other/", json={"jsonrpc":"2.0","id":"scope","method":"message/send","params":{"message":{"kind":"message","messageId":"bad","contextId":first.context_id,"role":"user","parts":[{"kind":"text","text":"bad"}]}}})
            assert mismatch.json()["error"]["code"] == -32602
    finally:
        await proxy._async_shutdown()


@pytest.mark.asyncio
async def test_two_concurrent_turns_complete_in_order(server):
    server.Context.started = asyncio.Event()
    server.Context.release = asyncio.Event()
    first, p1 = await submit(server, "slow")
    one = asyncio.create_task(server.worker.run_task(p1))
    await server.Context.started.wait()
    second, p2 = await submit(server, "second", first["context_id"])
    two = asyncio.create_task(server.worker.run_task(p2))
    await asyncio.sleep(0)
    assert second["status"]["state"] == "submitted"
    server.Context.release.set()
    await asyncio.gather(one, two)
    assert first["artifacts"][0]["parts"][0]["text"] == "slow"
    assert second["artifacts"][0]["parts"][0]["text"] == "slow / second"


@pytest.mark.asyncio
async def test_missing_native_context_and_failed_project_activation(server, monkeypatch):
    first, params = await submit(server, "first")
    await server.worker.run_task(params)
    entry = await server.storage.load_context(first["context_id"])
    server.Context.remove(entry.agent_context_id)
    with pytest.raises(server.adapter.ConversationUnavailable):
        await submit(server, "missing", first["context_id"])
    def fail(*args): raise RuntimeError("Project unavailable")
    monkeypatch.setattr(server.adapter.projects, "activate_project", fail)
    task, params = await submit(server, "must not run", project="invalid")
    await server.worker.run_task(params)
    entry = await server.storage.load_context(task["context_id"])
    assert task["status"]["state"] == "failed"
    assert entry.agent_context_id is None
    assert not server.Context.contexts


@pytest.mark.asyncio
async def test_header_auth_and_fragmented_project_request(server):
    proxy = server.adapter.DynamicA2AProxy()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy), base_url="http://test") as http:
            assert (await http.get("/.well-known/agent-card.json")).status_code == 401
            assert (await http.get("/.well-known/agent-card.json", headers={"x-api-key":"test-token"})).status_code == 200
            prefixed = await http.get("/a2a/.well-known/agent-card.json", headers={"x-api-key":"test-token"})
            assert prefixed.status_code == 200
            assert prefixed.json()["url"] == "http://test/a2a/"
            body = json.dumps({"jsonrpc":"2.0","id":"chunked","method":"message/send","params":{"message":{"kind":"message","messageId":"one","role":"user","parts":[{"kind":"text","text":"chunked"}],"metadata":{"project":"spoofed"}}}}).encode()
            async def chunks():
                for i in range(0, len(body), 7): yield body[i:i+7]
            response = await http.post("/a2a/t-test-token/p-sales/", content=chunks(), headers={"content-type":"application/json"})
            assert response.status_code == 200
            task = response.json()["result"]
            entry = await proxy._storage.load_context(task["contextId"])
            assert entry.project == "sales"
    finally:
        await proxy._async_shutdown()
