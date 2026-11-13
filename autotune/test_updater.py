"""Tests of the upgrade reminder (updater.py) on throwaway git repositories.

Run with:  QT_QPA_PLATFORM=offscreen poetry run pytest test_updater.py
"""

import pytest
import updater
from PyQt5.QtWidgets import QApplication


def git(cwd, *args):
    updater._git(cwd, "-c", "user.name=test", "-c", "user.email=test@test", *args)


def commit(repo, path, content, message):
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    git(repo, "add", path)
    git(repo, "commit", "-m", message)


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """The user's checkout and a publisher's checkout of the same remote."""
    monkeypatch.delenv("UV", raising=False)
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "--initial-branch=master", str(remote))
    publisher = tmp_path / "publisher"
    git(tmp_path, "clone", str(remote), str(publisher))
    commit(publisher, "autotune/autotune.py", "v1", "autotune: first")
    commit(publisher, "autotune/pyproject.toml", "deps1", "autotune: deps")
    commit(publisher, "other/tool.py", "v1", "other: first")
    git(publisher, "push", "origin", "master")
    user = tmp_path / "user"
    git(tmp_path, "clone", str(remote), str(user))
    return publisher, user / "autotune"


def publish(publisher, path, content, message):
    commit(publisher, path, content, message)
    git(publisher, "push", "origin", "master")


def test_no_update_when_up_to_date(repos):
    _, tool_dir = repos
    assert updater.findUpdate(tool_dir) is None


def test_finds_the_new_commits_of_the_tool(repos):
    publisher, tool_dir = repos
    publish(publisher, "autotune/autotune.py", "v2", "autotune: second")
    publish(publisher, "other/tool.py", "v2", "other: second")
    publish(publisher, "autotune/autotune.py", "v3", "autotune: third")

    update = updater.findUpdate(tool_dir)

    assert update.changes == ["autotune: third", "autotune: second"]


def test_ignores_the_changes_of_other_tools(repos):
    publisher, tool_dir = repos
    publish(publisher, "other/tool.py", "v2", "other: second")
    assert updater.findUpdate(tool_dir) is None


def test_no_update_on_a_branch_with_local_commits(repos):
    publisher, tool_dir = repos
    publish(publisher, "autotune/autotune.py", "v2", "autotune: second")
    commit(tool_dir.parent, "autotune/local.py", "x", "autotune: local work")
    assert updater.findUpdate(tool_dir) is None


def test_no_update_without_upstream_branch(repos):
    publisher, tool_dir = repos
    publish(publisher, "autotune/autotune.py", "v2", "autotune: second")
    git(tool_dir, "checkout", "--quiet", "-b", "feature", "--no-track")
    assert updater.findUpdate(tool_dir) is None


def test_no_update_outside_a_git_checkout(tmp_path):
    assert updater.findUpdate(tmp_path) is None


def test_upgrade_fast_forwards(repos):
    publisher, tool_dir = repos
    publish(publisher, "autotune/autotune.py", "v2", "autotune: second")
    updater.findUpdate(tool_dir)

    result = updater.upgrade(tool_dir)

    assert (tool_dir / "autotune.py").read_text() == "v2"
    assert not result.dependencies_changed
    assert updater.findUpdate(tool_dir) is None


def test_upgrade_syncs_changed_dependencies(repos, monkeypatch):
    publisher, tool_dir = repos
    publish(publisher, "autotune/pyproject.toml", "deps2", "autotune: new deps")
    updater.findUpdate(tool_dir)
    synced = []
    monkeypatch.setattr(updater, "syncDependencies", synced.append)

    result = updater.upgrade(tool_dir)

    assert result.dependencies_changed
    assert synced == [tool_dir]


def test_upgrade_keeps_conflicting_local_changes(repos):
    publisher, tool_dir = repos
    publish(publisher, "autotune/autotune.py", "v2", "autotune: second")
    updater.findUpdate(tool_dir)
    (tool_dir / "autotune.py").write_text("local edit")

    with pytest.raises(updater.UpgradeError):
        updater.upgrade(tool_dir)

    assert (tool_dir / "autotune.py").read_text() == "local edit"


def test_restart_through_uv_when_run_by_uv(monkeypatch, tmp_path):
    monkeypatch.setenv("UV", "/bin/uv")
    command = updater.restartCommand(["autotune.py", "-x"], tool_dir=tmp_path)
    assert command[:4] == ["/bin/uv", "run", "--project", str(tmp_path)]
    assert command[4].endswith("autotune.py") and command[5:] == ["-x"]


def test_banner_offers_upgrade_then_restart(monkeypatch):
    app = QApplication.instance() or QApplication([])
    banner = updater.UpdateBanner()
    assert banner.isHidden()

    banner.showUpdate(updater.Update(changes=["autotune: second"]))
    assert not banner.isHidden()
    assert "1 change)" in banner.label.text()
    assert banner.label.toolTip() == "• autotune: second"
    assert banner.btn_restart.isHidden()

    banner.showUpgraded(updater.UpgradeResult(dependencies_changed=False))
    assert banner.btn_upgrade.isHidden()
    assert not banner.btn_restart.isHidden()

    banner.showUpgraded(
        updater.UpgradeResult(dependencies_changed=True, sync_error="no poetry")
    )
    assert banner.label.text() == "Upgraded, but no poetry"
    assert banner.btn_restart.isHidden()
    app.processEvents()
