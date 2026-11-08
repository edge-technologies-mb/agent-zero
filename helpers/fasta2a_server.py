# noqa: D401 (docstrings) – internal helper
import asyncio
import uuid
import atexit
import json
from typing import Any, List
import contextlib
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from helpers import settings, projects
from starlette.responses import Response as StarletteResponse
from starlette.requests import Request

# Local imports
from helpers.print_style import PrintStyle
from agent import AgentContext, UserMessage, AgentContextType
from initialize import initialize_agent

# Import FastA2A
try:
    from fasta2a import Worker, FastA2A  # type: ignore
    from fasta2a.broker import InMemoryBroker  # type: ignore
    from fasta2a.storage import InMemoryStorage  # type: ignore
    from fasta2a.schema import Message, Artifact, AgentProvider, Skill  # type: ignore
    FASTA2A_AVAILABLE = True
except ImportError:  # pragma: no cover – library not installed
    FASTA2A_AVAILABLE = False
    # Minimal stubs for type checkers when FastA2A is not available

    class Worker:  # type: ignore
        def __init__(self, **kwargs):
            pass

        async def run_task(self, params):
            pass

        async def cancel_task(self, params):
            pass

        def build_message_history(self, history):
            return []

        def build_artifacts(self, result):
            return []

    class FastA2A:  # type: ignore
        def __init__(self, **kwargs):
            pass

        async def __call__(self, scope, receive, send):
            pass

    class InMemoryBroker:  # type: ignore
        pass

    class InMemoryStorage:  # type: ignore
        async def update_task(self, **kwargs):
            pass

    Message = Artifact = AgentProvider = Skill = Any  # type: ignore

_PRINTER = PrintStyle(italic=True, font_color="purple", padding=False)


def _enable_streaming_capability(agent_card_body: bytes) -> bytes:
    """Return an agent-card JSON body with A2A streaming enabled."""
    agent_card = json.loads(agent_card_body)
    capabilities = agent_card.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
        agent_card["capabilities"] = capabilities
    capabilities["streaming"] = True
    return json.dumps(agent_card, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ConversationUnavailable(ValueError):
    """A caller supplied a context that cannot be resumed in this scope."""


@dataclass
class _Conversation:
    project: str | None
    agent_context_id: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pending: int = 0
    active_task_id: str | None = None


class AgentZeroStorage(InMemoryStorage):
    """Process-local conversations and RFC 3339 task timestamps."""

    def _prune(self):
        now = datetime.now(timezone.utc)
        for context_id, entry in list(self.contexts.items()):
            if entry.pending:
                continue
            context = AgentContext.get(entry.agent_context_id) if entry.agent_context_id else None
            if (entry.agent_context_id and context is None) or now - entry.last_used >= timedelta(hours=24):
                if context:
                    context.reset()
                    AgentContext.remove(context.id)
                del self.contexts[context_id]

    async def submit_task(self, context_id, message):
        self._prune()
        project = (message.get('metadata') or {}).get('project') or None
        entry = await self.load_context(context_id)
        unavailable = entry is None or entry.project != project
        if entry and entry.agent_context_id and AgentContext.get(entry.agent_context_id) is None:
            unavailable = True
        if 'context_id' in message and unavailable:
            raise ConversationUnavailable('Conversation unavailable for this project')
        if entry is None:
            entry = _Conversation(project=project)
            await self.update_context(context_id, entry)
        task = await super().submit_task(context_id, message)
        task['status']['timestamp'] = datetime.now(timezone.utc).isoformat()
        entry.pending += 1
        return task

    async def update_task(self, task_id, state, new_artifacts=None, new_messages=None):
        # A cancelled task's eventual worker result must not resurrect it.
        current = self.tasks[task_id]
        if current['status']['state'] in ('completed', 'failed', 'canceled', 'rejected'):
            return current
        task = await super().update_task(task_id, state, new_artifacts, new_messages)
        task['status']['timestamp'] = datetime.now(timezone.utc).isoformat()
        return task

    async def load_task(self, task_id, history_length=None):
        # FastA2A 0.6.1 otherwise trims the canonical history during polling.
        task = self.tasks.get(task_id)
        if task is None:
            return None
        result = deepcopy(task)
        if history_length is not None:
            result['history'] = result.get('history', [])[-history_length:] if history_length else []
        return result

    def close_contexts(self):
        for entry in self.contexts.values():
            if entry.agent_context_id:
                context = AgentContext.get(entry.agent_context_id)
                if context:
                    context.reset()
                    AgentContext.remove(context.id)
        self.contexts.clear()


class AgentZeroFastA2A(FastA2A):
    """FastA2A with Agent Zero's routing and conversation errors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if FASTA2A_AVAILABLE:
            self.router.add_route('/.well-known/agent.json', self._agent_card_endpoint, methods=['HEAD', 'GET', 'OPTIONS'])

    async def _agent_run_endpoint(self, request: Request) -> StarletteResponse:
        try:
            return await super()._agent_run_endpoint(request)
        except ConversationUnavailable:
            data = await request.json()
            return StarletteResponse(content=json.dumps({
                'jsonrpc': '2.0', 'id': data.get('id'),
                'error': {'code': -32602, 'message': 'Conversation unavailable for this project'},
            }), media_type='application/json')

    async def _agent_card_endpoint(self, request: Request) -> StarletteResponse:
        response = await super()._agent_card_endpoint(request)
        body = json.loads(_enable_streaming_capability(response.body))
        # Render request-specific routing without poisoning the cached base card.
        body['url'] = request.scope.get('a0_a2a_base_url', str(request.base_url))
        return StarletteResponse(content=json.dumps(body), media_type='application/json')


class AgentZeroWorker(Worker):
    """Execute separate A2A tasks against a retained, serialized conversation."""

    def __init__(self, broker, storage):
        super().__init__(broker=broker, storage=storage)
        self.storage = storage

    async def run_task(self, params: Any) -> None:
        task_id = params['id']
        entry = await self.storage.load_context(params['context_id'])
        if entry is None:
            await self.storage.update_task(task_id, state='failed')
            return
        try:
            async with entry.lock:
                task = await self.storage.load_task(task_id)
                if task['status']['state'] == 'canceled':
                    return
                context = AgentContext.get(entry.agent_context_id) if entry.agent_context_id else None
                if entry.agent_context_id and context is None:
                    raise ConversationUnavailable('Conversation expired while queued')
                if context is None:
                    context = AgentContext(initialize_agent(), type=AgentContextType.BACKGROUND)
                    try:
                        context.set_data('lifetime_hours', 24)
                        if entry.project:
                            projects.activate_project(context.id, entry.project)
                    except Exception:
                        context.reset()
                        AgentContext.remove(context.id)
                        raise
                    entry.agent_context_id = context.id
                entry.active_task_id = task_id
                await self.storage.update_task(task_id, state='working')
                message = self._convert_message(params['message'])
                context.log.log(type='user', heading='Remote user message', content=message.message, kvps={'from': 'A2A'})
                result = await context.communicate(message).result()
                await self.storage.update_task(
                    task_id, state='completed',
                    new_messages=[{'role': 'agent', 'kind': 'message', 'message_id': str(uuid.uuid4()),
                                   'parts': [{'kind': 'text', 'text': str(result)}]}],
                    new_artifacts=self.build_artifacts(result),
                )
        except asyncio.CancelledError:
            if entry.active_task_id == task_id and entry.agent_context_id:
                context = AgentContext.get(entry.agent_context_id)
                if context:
                    context.kill_process()
            await self.storage.update_task(task_id, state='canceled')
            raise
        except Exception as exc:
            _PRINTER.print(f'[A2A] Task failed ({type(exc).__name__})')
            await self.storage.update_task(task_id, state='failed')
        finally:
            if entry.active_task_id == task_id:
                entry.active_task_id = None
            entry.pending -= 1
            entry.last_used = datetime.now(timezone.utc)
            if entry.agent_context_id:
                context = AgentContext.get(entry.agent_context_id)
                if context:
                    context.last_message = entry.last_used

    async def cancel_task(self, params: Any) -> None:
        task = await self.storage.load_task(params['id'])
        if task is None or task['status']['state'] in ('completed', 'failed', 'canceled', 'rejected'):
            return
        await self.storage.update_task(task['id'], state='canceled')
        entry = await self.storage.load_context(task['context_id'])
        if entry and entry.active_task_id == task['id'] and entry.agent_context_id:
            context = AgentContext.get(entry.agent_context_id)
            if context:
                context.kill_process()

    def build_message_history(self, history: List[Any]) -> List[Message]:  # type: ignore
        # Not used in this simplified implementation
        return []

    def build_artifacts(self, result: Any) -> List[Artifact]:  # type: ignore
        return [{'artifact_id': str(uuid.uuid4()), 'parts': [{'kind': 'text', 'text': str(result)}]}]

    def _convert_message(self, a2a_message: Message) -> UserMessage:  # type: ignore
        """Convert A2A message to Agent Zero UserMessage."""
        # Extract text from message parts
        text_parts = [part.get('text', '') for part in a2a_message.get('parts', []) if part.get('kind') == 'text']
        message_text = '\n'.join(text_parts)

        # Extract file attachments
        attachments = []
        for part in a2a_message.get('parts', []):
            if part.get('kind') == 'file':
                file_info = part.get('file', {})
                if 'uri' in file_info:
                    attachments.append(file_info['uri'])

        return UserMessage(
            message=message_text,
            attachments=attachments
        )


class DynamicA2AProxy:
    """Dynamic proxy for FastA2A server that allows reconfiguration."""

    _instance = None

    def __init__(self):
        self.app = None
        self.token = ""
        self._lock = threading.Lock()  # Use threading.Lock instead of asyncio.Lock
        self._startup_done: bool = False
        self._worker_bg_task: asyncio.Task | None = None
        self._reconfigure_needed: bool = False  # Flag for deferred reconfiguration

        if FASTA2A_AVAILABLE:
            # Initialize with default token
            cfg = settings.get_settings()
            self.token = cfg.get("mcp_server_token", "")
            self._configure()
            self._register_shutdown()
        else:
            _PRINTER.print("[A2A] FastA2A not available, server will return 503")

    @staticmethod
    def get_instance():
        if DynamicA2AProxy._instance is None:
            DynamicA2AProxy._instance = DynamicA2AProxy()
        return DynamicA2AProxy._instance

    def reconfigure(self, token: str):
        """Reconfigure the FastA2A server with new token."""
        self.token = token
        if FASTA2A_AVAILABLE:
            with self._lock:
                # Mark that reconfiguration is needed - will be done on next request
                self._reconfigure_needed = True
                self._startup_done = False  # Force restart on next request
                _PRINTER.print("[A2A] Reconfiguration scheduled for next request")

    def _configure(self):
        """Configure the FastA2A application with Agent Zero integration."""
        try:
            storage = AgentZeroStorage()  # type: ignore[arg-type]
            broker = InMemoryBroker()  # type: ignore[arg-type]

            # Define Agent Zero's skills
            skills: List[Skill] = [{  # type: ignore
                "id": "general_assistance",
                "name": "General AI Assistant",
                "description": "Provides general AI assistance including code execution, file management, web browsing, and problem solving",
                "tags": ["ai", "assistant", "code", "files", "web", "automation"],
                "examples": [
                    "Write and execute Python code",
                    "Manage files and directories",
                    "Browse the web and extract information",
                    "Solve complex problems step by step",
                    "Install software and manage systems"
                ],
                "input_modes": ["text/plain", "application/octet-stream"],
                "output_modes": ["text/plain", "application/json"]
            }]

            provider: AgentProvider = {  # type: ignore
                "organization": "Agent Zero",
                "url": "https://github.com/frdel/agent-zero"
            }

            # Create new FastA2A app with proper thread safety
            new_app = AgentZeroFastA2A(  # type: ignore
                storage=storage,
                broker=broker,
                name="Agent Zero",
                description=(
                    "A general AI assistant that can execute code, manage files, browse the web, and "
                    "solve complex problems in an isolated Linux environment."
                ),
                version="1.0.0",
                provider=provider,
                skills=skills,
                lifespan=None,  # We manage lifespan manually
                middleware=[],  # No middleware - we handle auth in wrapper
            )

            # Store for later lazy startup (needs active event-loop)
            self._storage = storage  # type: ignore[attr-defined]
            self._broker = broker  # type: ignore[attr-defined]
            self._worker = AgentZeroWorker(broker=broker, storage=storage)  # type: ignore[attr-defined]

            # Atomic update of the app
            self.app = new_app

            # _PRINTER.print("[A2A] FastA2A server configured successfully")

        except Exception as e:
            _PRINTER.print(f"[A2A] Failed to configure FastA2A server: {e}")
            self.app = None
            raise

    # ---------------------------------------------------------------------
    # Shutdown handling
    # ---------------------------------------------------------------------

    def _register_shutdown(self):
        """Register an atexit hook to gracefully stop worker & task manager."""

        def _sync_shutdown():
            try:
                if not self._startup_done or not FASTA2A_AVAILABLE:
                    return
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._async_shutdown())
                loop.close()
            except Exception:
                pass  # ignore errors during interpreter shutdown

        atexit.register(_sync_shutdown)

    async def _async_shutdown(self):
        """Async shutdown: cancel worker task & close task manager."""
        if self._worker_bg_task and not self._worker_bg_task.done():
            self._worker_bg_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_bg_task
        try:
            if hasattr(self, 'app') and self.app:
                await self.app.task_manager.__aexit__(None, None, None)  # type: ignore[attr-defined]
        except Exception:
            pass

        if hasattr(self, '_storage'):
            self._storage.close_contexts()

    async def _async_reconfigure(self):
        """Perform async reconfiguration with proper lifecycle management."""
        _PRINTER.print("[A2A] Starting async reconfiguration")

        # Shutdown existing components
        await self._async_shutdown()

        # Reset startup state
        self._startup_done = False
        self._worker_bg_task = None

        # Reconfigure with new token
        self._configure()

        # Restart components
        await self._startup()

        # Clear reconfiguration flag
        self._reconfigure_needed = False

        _PRINTER.print("[A2A] Async reconfiguration completed")

    async def _startup(self):
        """Ensure TaskManager and Worker are running inside current event-loop."""
        if self._startup_done or not FASTA2A_AVAILABLE:
            return
        self._startup_done = True

        # Start task manager
        await self.app.task_manager.__aenter__()  # type: ignore[attr-defined]

        async def _worker_loop():
            async with self._worker.run():  # type: ignore[attr-defined]
                await asyncio.Event().wait()

        # fire-and-forget background task – keep reference
        self._worker_bg_task = asyncio.create_task(_worker_loop())
        _PRINTER.print("[A2A] Worker & TaskManager started")

    async def __call__(self, scope, receive, send):
        """ASGI application interface with token-based routing."""
        if not FASTA2A_AVAILABLE:
            # FastA2A not available, return 503
            response = b'HTTP/1.1 503 Service Unavailable\r\n\r\nFastA2A not available'
            await send({
                'type': 'http.response.start',
                'status': 503,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': response,
            })
            return

        from helpers import settings
        cfg = settings.get_settings()
        if not cfg["a2a_server_enabled"]:
            response = b'HTTP/1.1 403 Forbidden\r\n\r\nA2A server is disabled'
            await send({
                'type': 'http.response.start',
                'status': 403,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': response,
            })
            return

        # Check if reconfiguration is needed
        if self._reconfigure_needed:
            try:
                await self._async_reconfigure()
            except Exception as e:
                _PRINTER.print(f"[A2A] Error during reconfiguration: {e}")
                # Return 503 if reconfiguration failed
                await send({
                    'type': 'http.response.start',
                    'status': 503,
                    'headers': [[b'content-type', b'text/plain']],
                })
                await send({
                    'type': 'http.response.body',
                    'body': b'FastA2A reconfiguration failed',
                })
                return

        if self.app is None:
            # FastA2A not configured, return 503
            response = b'HTTP/1.1 503 Service Unavailable\r\n\r\nFastA2A not configured'
            await send({
                'type': 'http.response.start',
                'status': 503,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': response,
            })
            return

        # Lazy-start background components the first time we get a request
        if not self._startup_done:
            try:
                _PRINTER.print("[A2A] Starting up FastA2A components")
                await self._startup()
            except Exception as e:
                _PRINTER.print(f"[A2A] Error during startup: {e}")
                # Return 503 if startup failed
                await send({
                    'type': 'http.response.start',
                    'status': 503,
                    'headers': [[b'content-type', b'text/plain']],
                })
                await send({
                    'type': 'http.response.body',
                    'body': b'FastA2A startup failed',
                })
                return

        # Preserve the externally visible route before stripping authentication/project prefixes.
        original_path = scope.get('path', '')
        base_path = original_path.split('/.well-known/', 1)[0].rstrip('/') + '/'
        from starlette.datastructures import URL
        external_url = str(URL(scope=scope).replace(path=base_path, query=''))

        # Handle token-based routing: /a2a/t-{token}/... or /t-{token}/...
        path = scope.get('path', '')

        # Strip /a2a prefix if present (DispatcherMiddleware doesn't always strip it)
        if path.startswith('/a2a'):
            path = path[4:]  # Remove '/a2a' prefix

        # Initialize project name
        project_name = None

        # Check if path matches token pattern /t-{token}/
        if path.startswith('/t-'):
            # Extract token from path
            if '/' in path[3:]:
                path_parts = path[3:].split('/', 1)  # Remove '/t-' prefix
                request_token = path_parts[0]
                remaining_path = '/' + path_parts[1] if len(path_parts) > 1 else '/'

                # Check for project pattern /p-{project}/
                if remaining_path.startswith('/p-'):
                    project_parts = remaining_path[3:].split('/', 1)
                    if project_parts[0]:
                        project_name = project_parts[0]
                        remaining_path = '/' + project_parts[1] if len(project_parts) > 1 else '/'
                        _PRINTER.print(f"[A2A] Extracted project from URL: {project_name}")
            else:
                request_token = path[3:]
                remaining_path = '/'

            # Validate token
            cfg = settings.get_settings()
            expected_token = cfg.get("mcp_server_token")

            if expected_token and request_token != expected_token:
                # Invalid token, return 401
                await send({
                    'type': 'http.response.start',
                    'status': 401,
                    'headers': [[b'content-type', b'text/plain']],
                })
                await send({
                    'type': 'http.response.body',
                    'body': b'Unauthorized',
                })
                return

            # If project specified, inject it into the request payload
            if project_name:
                # Buffer messages and modify before returning the complete body
                received_messages = []
                body_modified = False
                original_receive = receive

                async def receive_wrapper():
                    nonlocal body_modified
                    if body_modified:
                        return await original_receive()
                    # ASGI may split a request into several chunks. Do not emit the
                    # original prefix before emitting the rewritten complete body.
                    while True:
                        message = await original_receive()
                        if message['type'] != 'http.request':
                            return message
                        received_messages.append(message)
                        if not message.get('more_body', False):
                            break
                    body_modified = True
                    full_body = b''.join(msg.get('body', b'') for msg in received_messages)
                    try:
                        data = json.loads(full_body)
                        if 'params' in data and 'message' in data['params']:
                            msg_data = data['params']['message']
                            if not isinstance(msg_data.get('metadata'), dict):
                                msg_data['metadata'] = {}
                            msg_data['metadata']['project'] = project_name
                        full_body = json.dumps(data).encode('utf-8')
                    except (ValueError, TypeError):
                        pass  # Let the protocol parser reject invalid input.
                    return {'type': 'http.request', 'body': full_body, 'more_body': False}

                receive = receive_wrapper

            # Update scope with cleaned path
            scope = dict(scope)
            scope['path'] = remaining_path
            scope['a0_a2a_base_url'] = external_url
        else:
            # No token in path, check other auth methods
            request = Request(scope, receive=receive)

            cfg = settings.get_settings()
            expected = cfg.get("mcp_server_token")

            if expected:
                auth_header = request.headers.get("Authorization", "")
                api_key = request.headers.get("X-API-KEY") or request.query_params.get("api_key")

                is_authorized = (
                    (auth_header.startswith("Bearer ") and auth_header.split(" ", 1)[1] == expected) or
                    (api_key == expected)
                )

                if not is_authorized:
                    # No valid auth, return 401
                    await send({
                        'type': 'http.response.start',
                        'status': 401,
                        'headers': [[b'content-type', b'text/plain']],
                    })
                    await send({
                        'type': 'http.response.body',
                        'body': b'Unauthorized',
                    })
                    return
            else:
                _PRINTER.print("[A2A] No expected token found in settings")

        scope = dict(scope)
        scope.setdefault('a0_a2a_base_url', external_url)
        if not path.startswith('/t-'):
            scope['path'] = path or '/'
        # Delegate to FastA2A app with cleaned scope
        with self._lock:
            app = self.app
        if app:
            await app(scope, receive, send)
        else:
            # App not configured, return 503
            await send({
                'type': 'http.response.start',
                'status': 503,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': b'FastA2A app not configured',
            })
            return


def is_available():
    """Check if FastA2A is available and properly configured."""
    return FASTA2A_AVAILABLE and DynamicA2AProxy.get_instance().app is not None


def get_proxy():
    """Get the FastA2A proxy instance."""
    return DynamicA2AProxy.get_instance()
