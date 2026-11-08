"""Offline regression coverage for repository-aware update discovery and execution."""
import subprocess
from pathlib import Path

import pytest

from test_self_update_tag_filter import load_self_update_manager, self_update


OFFICIAL = "https://github.com/agent0ai/agent-zero.git"
FORK = "https://github.com/example/agent-zero.git"


@pytest.fixture(autouse=True)
def isolated_source(monkeypatch):
    monkeypatch.delenv("A0_SELF_UPDATE_REMOTE_URL", raising=False)
    monkeypatch.setattr(self_update, "_remote_branch_list_cache", None)
    monkeypatch.setattr(self_update, "_remote_branch_list_cache_source", None)
    monkeypatch.setattr(self_update, "_remote_branch_tag_cache", {})
    monkeypatch.setattr(self_update, "_remote_branch_head_cache", {})


@pytest.fixture
def fork(monkeypatch):
    monkeypatch.setenv("A0_SELF_UPDATE_REMOTE_URL", FORK)
    return load_self_update_manager()


def test_official_source_remains_default():
    assert self_update._get_update_remote_url() == OFFICIAL
    assert not self_update.is_custom_repository()
    load_self_update_manager().validate_update_repository({})  # Legacy official requests.


@pytest.mark.parametrize("url", ["http://example.invalid/repo", "https://token@example.invalid/repo", "https://example.invalid/repo?token=secret", "https://example.invalid/repo#fragment", ""])
def test_invalid_configured_source_rejected(monkeypatch, url):
    monkeypatch.setenv("A0_SELF_UPDATE_REMOTE_URL", url)
    with pytest.raises(ValueError, match="HTTPS URL without credentials"):
        self_update._get_update_remote_url()
    manager = load_self_update_manager()
    with pytest.raises(ValueError, match="HTTPS URL without credentials"):
        manager.validate_update_repository({"remote_url": url})


def test_branch_cache_is_scoped_to_repository(monkeypatch):
    calls = []
    def run(*args):
        calls.append(args)
        return "abc refs/heads/main" if args[-1] == OFFICIAL else "def refs/heads/sales"
    monkeypatch.setattr(self_update, "_run_git_raw", run)
    assert self_update._get_remote_branch_names() == ["main"]
    monkeypatch.setenv("A0_SELF_UPDATE_REMOTE_URL", FORK)
    assert self_update._get_remote_branch_names() == ["sales"]
    assert self_update._get_remote_branch_names() == ["sales"]
    assert len(calls) == 2
    assert calls[-1][-1] == FORK


@pytest.mark.parametrize("method, fallback", [
    ("get_available_branch_values", "_get_local_origin_branch_names"),
    ("_get_branch_merged_tags", "_get_local_branch_merged_tags"),
    ("_get_branch_head_info", "_get_local_branch_head_info"),
])
def test_unreachable_fork_never_falls_back_to_origin(monkeypatch, fork, method, fallback):
    def fail(*args, **kwargs):
        raise RuntimeError("offline")
    def forbidden(*args, **kwargs):
        pytest.fail("Fork discovery must not fall back to origin")
    monkeypatch.setattr(self_update, "_run_git_raw", fail)
    monkeypatch.setattr(self_update, "_run_git", fail)
    monkeypatch.setattr(self_update, fallback, forbidden)
    args = () if method == "get_available_branch_values" else ("main",)
    with pytest.raises(RuntimeError, match="configured update repository"):
        getattr(self_update, method)(*args)


@pytest.fixture
def git_fixture(tmp_path, monkeypatch, fork):
    source = tmp_path / "source"
    installed = tmp_path / "installed"
    source.mkdir()
    installed.mkdir()
    def git(repo, *args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
    git(source, "init", "-q", "-b", "main")
    git(source, "config", "user.name", "Updater test")
    git(source, "config", "user.email", "updater@example.invalid")
    (source / "version.txt").write_text("release")
    git(source, "add", ".")
    git(source, "commit", "-qm", "Release")
    git(source, "tag", "v2.15")
    (source / "version.txt").write_text("fork patch")
    git(source, "commit", "-qam", "Fork patch")
    head = git(source, "rev-parse", "HEAD")
    git(installed, "init", "-q")
    original_run = self_update._run_git
    def local_run(repo, *args):
        if args[0] == "fetch":
            assert FORK in args
        return original_run(repo, *(str(source) if arg == FORK else arg for arg in args))
    monkeypatch.setattr(self_update, "_run_git", local_run)
    monkeypatch.setattr(self_update, "durable_self_update_supports_latest", lambda repo_dir=None: True)
    def no_releases(*args):
        pytest.fail("A fork must not use official GitHub releases")
    monkeypatch.setattr(self_update.git, "get_remote_releases", no_releases)
    original_command = fork.run_command
    def command(args, **kwargs):
        if "fetch" in args:
            assert FORK in args
        return original_command([str(source) if arg == FORK else arg for arg in args], **kwargs)
    monkeypatch.setattr(fork, "run_command", command)
    return source, installed, head, git


def test_fork_discovery_and_execution_use_branch_head(monkeypatch, fork, git_fixture):
    source, installed, head, git = git_fixture
    tags, error = self_update.get_available_tags("main", repo_dir=installed)
    assert (tags, error) == (["v2.15"], "")
    options, majors, error = self_update.get_selector_tag_options("main", repo_dir=installed, current_version="v2.15")
    assert options[0] == {"value": "latest", "label": "latest (v2.15+1)"}
    assert majors == [] and error == ""
    resolved = fork.resolve_requested_target(installed, "main", "latest", "v2.15", fork.NullLogger())
    assert resolved["expected_commit"] == head
    assert resolved["target_ref"] == "refs/remotes/a0-self-update/main"
    explicit = fork.resolve_requested_target(installed, "main", "v2.15", "v2.15", fork.NullLogger())
    assert explicit["expected_commit"] == git(source, "rev-parse", "v2.15")
    assert explicit["expected_commit"] != head


def test_fork_latest_keeps_major_guard(monkeypatch, fork, git_fixture):
    source, installed, head, git = git_fixture
    git(source, "tag", "v3.0")
    options, majors, _ = self_update.get_selector_tag_options("main", repo_dir=installed, current_version="v2.15")
    assert "latest" not in [option["value"] for option in options]
    assert majors == [3]
    with pytest.raises(RuntimeError, match="major versions"):
        fork.resolve_requested_target(installed, "main", "latest", "v2.15", fork.NullLogger())


def test_fork_rejects_tag_outside_selected_branch(fork, git_fixture):
    source, installed, head, git = git_fixture
    git(source, "checkout", "-qb", "other")
    (source / "other.txt").write_text("other branch")
    git(source, "add", ".")
    git(source, "commit", "-qm", "Other release")
    git(source, "tag", "v2.16")
    with pytest.raises(RuntimeError, match="not reachable"):
        fork.resolve_requested_target(installed, "main", "v2.16", "v2.15", fork.NullLogger())


@pytest.mark.parametrize("method, cache", [("_get_remote_branch_head_info", "_remote_branch_head_cache"), ("_get_remote_branch_merged_tags", "_remote_branch_tag_cache")])
def test_branch_detail_caches_are_repository_scoped(monkeypatch, method, cache):
    calls = []
    def run(repo, *args):
        if args[0] == "fetch":
            calls.append(args[3])
        if args[0] == "describe":
            return "v2.15"
        if args[0] == "rev-parse":
            return "abc123"
        if args[0] == "tag":
            return "v2.15"
        return ""
    monkeypatch.setattr(self_update, "_run_git", run)
    monkeypatch.setattr(self_update, "_get_tag_release_time_in_repo", lambda *args: "")
    getattr(self_update, method)("main")
    monkeypatch.setenv("A0_SELF_UPDATE_REMOTE_URL", FORK)
    getattr(self_update, method)("main")
    getattr(self_update, method)("main")
    assert len(calls) == 2
    assert len(getattr(self_update, cache)) == 2


def test_fork_update_info_defaults_to_main_latest(monkeypatch, fork, tmp_path):
    version = {"branch": "main-edge-tech", "short_tag": "v2.15", "describe": "v2.15-1-gabc", "commit": "abc"}
    monkeypatch.setattr(self_update, "get_repo_version_info", lambda *args: version)
    monkeypatch.setattr(self_update, "get_available_branches", lambda **kwargs: [{"value": "main", "label": "main"}])
    monkeypatch.setattr(self_update, "get_selector_tag_options", lambda *args, **kwargs: ([{"value": "latest", "label": "latest (v2.15+1)"}], [], ""))
    monkeypatch.setattr(self_update, "get_current_branch_latest_info", lambda branch, **kwargs: {"branch": branch})
    monkeypatch.setattr(self_update, "load_pending_update", lambda: None)
    monkeypatch.setattr(self_update, "load_last_status", lambda: None)
    info = self_update.get_update_info(tmp_path)
    assert info["repository"] == {"url": FORK, "is_custom": True}
    assert info["defaults"]["branch"] == "main"
    assert info["defaults"]["tag"] == "latest"
    assert info["main_branch_latest"]["branch"] == "main"


def test_schedule_binds_request_to_configured_source(monkeypatch, fork, tmp_path):
    monkeypatch.setattr(self_update, "get_repo_version_info", lambda *args: {"short_tag": "v2.15", "describe": "v2.15", "commit": "abc"})
    monkeypatch.setattr(self_update, "get_available_branch_values", lambda **kwargs: ["main"])
    monkeypatch.setattr(self_update, "durable_self_update_supports_repository", lambda *args: True)
    monkeypatch.setattr(self_update, "durable_self_update_supports_latest", lambda **kwargs: True)
    monkeypatch.setattr(self_update, "get_selector_tag_options", lambda *args, **kwargs: ([{"value": "latest"}], [], ""))
    request_file = tmp_path / "request.yaml"
    monkeypatch.setattr(self_update, "get_update_file_path", lambda: request_file)
    payload = self_update.schedule_update(branch="main", tag="latest", backup_usr=True, backup_path=str(tmp_path), backup_name="test.zip", backup_conflict_policy="rename", repo_dir=tmp_path)
    assert payload["remote_url"] == FORK
    assert self_update._load_yaml(request_file)["remote_url"] == FORK
    fork.validate_update_repository(payload)


@pytest.mark.parametrize("requested", [None, OFFICIAL, "https://example.invalid/other.git"])
def test_fork_rejects_unbound_or_changed_repository(fork, requested):
    with pytest.raises(ValueError, match="repository changed or was not validated"):
        fork.validate_update_repository({"remote_url": requested} if requested else {})


def test_fork_accepts_equivalent_repository_suffix(fork):
    fork.validate_update_repository({"remote_url": FORK.removesuffix(".git") + "/"})


def test_matching_tag_name_does_not_skip_fork_update(fork):
    assert not fork.installed_target_matches_request({"branch": "main", "describe": "v2.15"}, requested_branch="main", requested_tag="v2.15")


def test_fork_requires_durable_repository_support(monkeypatch, fork, tmp_path):
    stale = tmp_path / "updater.py"
    stale.write_text("# old updater")
    monkeypatch.setattr(self_update, "get_durable_self_update_manager_path", lambda: stale)
    assert not self_update.durable_self_update_supports_repository()
    monkeypatch.setattr(self_update, "get_repo_version_info", lambda *args: {"short_tag": "v2.15"})
    with pytest.raises(RuntimeError, match="fork-aware runtime upgrade"):
        self_update.schedule_update(branch="main", tag="latest", backup_usr=True, backup_path="", backup_name="", backup_conflict_policy="rename", repo_dir=tmp_path)


def test_repository_mismatch_is_recorded_before_checkout(monkeypatch, fork):
    events = []
    monkeypatch.setattr(fork, "get_repo_version_info", lambda *args: {"short_tag": "v2.15", "commit": "abc"})
    def forbidden(*args, **kwargs):
        pytest.fail("Repository mismatch must not mutate the checkout")
    for method in ("create_rollback_stash", "create_usr_backup", "resolve_requested_target", "checkout_target_release"):
        monkeypatch.setattr(fork, method, forbidden)
    monkeypatch.setattr(fork, "record_result", lambda **kwargs: events.append(kwargs))
    process = object()
    monkeypatch.setattr(fork, "launch_ui_process", lambda *args: process)
    assert fork.execute_pending_update({"branch": "main", "tag": "latest", "remote_url": OFFICIAL}, logger=fork.NullLogger()) is process
    assert events[0]["status"] == "failed"
    assert "repository changed" in events[0]["error"]


def test_cli_queue_binds_repository(monkeypatch, fork, tmp_path):
    monkeypatch.setattr(fork, "TRIGGER_FILE", tmp_path / "request.yaml")
    monkeypatch.setattr(fork, "get_repo_version_info", lambda *args: {"short_tag": "v2.15", "describe": "v2.15", "commit": "abc"})
    payload = fork.queue_update_request(branch="main", tag="latest", backup_usr=True, backup_path=str(tmp_path), backup_name="test.zip", backup_conflict_policy="rename")
    assert payload["remote_url"] == FORK
    fork.validate_update_repository(payload)


@pytest.mark.parametrize("healthy", [True, False])
def test_fork_update_preserves_data_and_rolls_back_failed_health(monkeypatch, fork, git_fixture, healthy):
    source, installed, head, git = git_fixture
    git(installed, "fetch", "-q", "--tags", str(source), "main")
    git(installed, "checkout", "-qb", "main", "v2.15")
    git(installed, "config", "user.name", "Updater test")
    git(installed, "config", "user.email", "updater@example.invalid")
    original = git(installed, "rev-parse", "HEAD")
    (installed / "version.txt").write_text("local change")
    (installed / ".git/info/exclude").write_text("usr/\n")
    (installed / "usr").mkdir()
    (installed / "usr/data.txt").write_text("preserved user data")
    monkeypatch.setattr(fork, "REPO_DIR", installed)
    monkeypatch.setattr(fork, "launch_ui_process", lambda *args: object())
    monkeypatch.setattr(fork, "terminate_process", lambda *args: None)
    monkeypatch.setattr(fork, "refresh_codex_cli", lambda *args: None)
    health = iter([healthy, True])
    monkeypatch.setattr(fork, "wait_for_health", lambda *args, **kwargs: (next(health), "fixture health"))
    outcomes = []
    monkeypatch.setattr(fork, "record_result", lambda **kwargs: outcomes.append(kwargs))
    request = {"branch": "main", "tag": "latest", "remote_url": FORK,
               "backup_usr": True, "backup_path": str(installed.parent / "backups"),
               "backup_name": "usr.zip", "backup_conflict_policy": "rename"}
    fork.execute_pending_update(request, logger=fork.NullLogger())
    assert git(installed, "rev-parse", "HEAD") == (head if healthy else original)
    assert (installed / "version.txt").read_text() == ("fork patch" if healthy else "local change")
    assert (installed / "usr/data.txt").read_text() == "preserved user data"
    assert git(installed, "stash", "list") == ""
    assert outcomes[-1]["status"] == ("success" if healthy else "rolled_back")
    assert Path(outcomes[-1]["backup_zip_path"]).is_file()


def test_changed_source_cannot_skip_as_already_installed(monkeypatch):
    manager = load_self_update_manager()
    request = {"branch": "main", "tag": "v2.15", "remote_url": FORK}
    monkeypatch.setattr(manager, "load_request_file", lambda: (request, "fixture"))
    monkeypatch.setattr(manager, "AttemptLogger", lambda *args: manager.NullLogger())
    monkeypatch.setattr(manager, "clean_uv_cache", lambda *args: None)
    monkeypatch.setattr(manager, "clean_transient_desktop_agent_state", lambda *args: None)
    monkeypatch.setattr(manager, "get_repo_version_info", lambda *args: {"branch": "main", "describe": "v2.15", "short_tag": "v2.15"})
    processed = []
    monkeypatch.setattr(manager, "execute_pending_update", lambda data, **kwargs: processed.append(data))
    monkeypatch.setattr(manager, "wait_for_process", lambda *args: 0)
    assert manager.docker_run_ui() == 0
    assert processed == [request]
