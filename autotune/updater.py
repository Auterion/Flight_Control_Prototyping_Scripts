"""Upgrade reminder: new commits of the tool on the branch the checkout tracks.

The tool is run from a git checkout, so an upgrade is a fast-forward of that
checkout, followed by a dependency sync when the dependencies changed.
"""

import os
import shutil

# Runs git and poetry with fixed arguments, never through a shell
import subprocess  # nosec B404
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

kToolDir = Path(__file__).resolve().parent
kDependencyFiles = ("pyproject.toml", "poetry.lock", "uv.lock")
kMaxListedChanges = 20
kGitTimeout = 30.0
kInstallTimeout = 600.0


class UpgradeError(Exception):
    pass


@dataclass
class Update:
    # Subjects of the new commits that change the tool, newest first
    changes: list


@dataclass
class UpgradeResult:
    dependencies_changed: bool
    # Error of the dependency sync, None if it succeeded or was not needed
    sync_error: str = None


def _run(program, args, cwd, timeout, env=None):
    """Standard output of program, UpgradeError if it fails."""
    executable = shutil.which(program)
    if executable is None:
        raise UpgradeError(f"{program} not found")
    try:
        result = subprocess.run(  # nosec B603
            [executable, *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise UpgradeError(str(e)) from e
    if result.returncode != 0:
        raise UpgradeError(result.stderr.strip() or f"{program} {args[0]} failed")
    return result.stdout.strip()


def _git(tool_dir, *args, timeout=kGitTimeout):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    # An SSH key passphrase prompt would hang the check: fail instead
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    return _run("git", args, tool_dir, timeout, env)


def findUpdate(tool_dir=kToolDir, fetch=True):
    """The commits of the upstream branch that change the tool, None if the
    checkout cannot be fast-forwarded to it or is not behind."""
    try:
        _git(tool_dir, "rev-parse", "--abbrev-ref", "@{upstream}")
        if fetch:
            _git(tool_dir, "fetch", "--quiet")
        ahead, behind = _git(
            tool_dir, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
        ).split()
        if int(ahead) > 0 or int(behind) == 0:
            return None
        log = _git(
            tool_dir,
            "log",
            "--format=%s",
            f"--max-count={kMaxListedChanges}",
            "HEAD..@{upstream}",
            "--",
            ".",
        )
    except UpgradeError:
        return None
    if not log:
        return None
    return Update(changes=log.splitlines())


def isRunByUv():
    return "UV" in os.environ


def upgrade(tool_dir=kToolDir):
    """Fast-forwards the checkout to its upstream branch and syncs the
    dependencies if they changed. Raises UpgradeError if git refuses."""
    old_head = _git(tool_dir, "rev-parse", "HEAD")
    _git(tool_dir, "merge", "--ff-only", "--quiet", "@{upstream}")
    changed = _git(
        tool_dir, "diff", "--name-only", old_head, "HEAD", "--", *kDependencyFiles
    )
    result = UpgradeResult(dependencies_changed=bool(changed))
    if result.dependencies_changed:
        result.sync_error = syncDependencies(tool_dir)
    return result


def syncDependencies(tool_dir=kToolDir):
    """Installs the dependencies of the new version, returns an error message
    or None."""
    if isRunByUv():
        # uv run syncs the environment when the tool restarts through it
        return None
    try:
        _run("poetry", ["install"], tool_dir, kInstallTimeout)
    except UpgradeError as e:
        return f"`poetry install` failed: {e}"
    return None


def restartCommand(argv=None, tool_dir=kToolDir):
    argv = sys.argv if argv is None else argv
    script = os.path.abspath(argv[0])
    if isRunByUv():
        uv = os.environ["UV"]
        return [uv, "run", "--project", str(tool_dir), script, *argv[1:]]
    return [sys.executable, script, *argv[1:]]


def restart(*extra_args):
    command = [*restartCommand(), *extra_args]
    # Replaces the tool by its new version
    os.execv(command[0], command)  # nosec B606


class UpdateBanner(QFrame):
    """Hidden until an update is found; then offers to upgrade and restart,
    the restart being left to restartRequested."""

    found = pyqtSignal(object)
    upgraded = pyqtSignal(object)
    failed = pyqtSignal(str)
    restartRequested = pyqtSignal()

    def __init__(self, parent=None, tool_dir=kToolDir):
        super().__init__(parent)
        self.tool_dir = tool_dir
        self.setObjectName("updateBanner")
        self.setStyleSheet(
            "#updateBanner { background: #e8f1fc; border: 1px solid #2a78d6;"
            " border-radius: 4px; }"
        )
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.btn_upgrade = QPushButton("Upgrade")
        self.btn_upgrade.clicked.connect(self.startUpgrade)
        self.btn_restart = QPushButton("Restart")
        self.btn_restart.clicked.connect(self.restartRequested)
        self.btn_later = QPushButton("Later")
        self.btn_later.clicked.connect(self.hide)
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(self.label, stretch=1)
        for button in (self.btn_upgrade, self.btn_restart, self.btn_later):
            button.setAutoDefault(False)
            layout.addWidget(button)
        self.setLayout(layout)
        self.found.connect(self.showUpdate)
        self.upgraded.connect(self.showUpgraded)
        self.failed.connect(self.showFailure)
        self.hide()

    def startCheck(self):
        # A daemon thread does not keep the tool open while a fetch hangs
        threading.Thread(target=self._check, daemon=True).start()

    def _check(self):
        update = findUpdate(self.tool_dir)
        if update is not None:
            self._emit(self.found, update)

    def _emit(self, signal, value):
        try:
            signal.emit(value)
        except RuntimeError:
            pass  # The window closed meanwhile

    def showUpdate(self, update):
        count = len(update.changes)
        more = "+" if count >= kMaxListedChanges else ""
        self.label.setText(
            f"A new version of autotune is available ({count}{more} "
            f"change{'s' if count > 1 else ''})."
        )
        self.label.setToolTip("\n".join(f"• {change}" for change in update.changes))
        self.btn_upgrade.setVisible(True)
        self.btn_upgrade.setEnabled(True)
        self.btn_restart.setVisible(False)
        self.btn_later.setEnabled(True)
        self.show()

    def startUpgrade(self):
        self.label.setText("Upgrading…")
        self.btn_upgrade.setEnabled(False)
        self.btn_later.setEnabled(False)
        # Not a daemon: closing the tool waits for a dependency sync to finish
        threading.Thread(target=self._upgrade).start()

    def _upgrade(self):
        try:
            result = upgrade(self.tool_dir)
        except UpgradeError as e:
            self._emit(self.failed, f"Upgrade failed: {e}")
        else:
            self._emit(self.upgraded, result)

    def showUpgraded(self, result):
        if result.sync_error:
            self.showFailure(f"Upgraded, but {result.sync_error}")
            return
        self.label.setText(
            "Upgraded. Restart to use the new version: the current session is "
            "restored."
        )
        self.label.setToolTip("")
        self.btn_upgrade.setVisible(False)
        self.btn_restart.setVisible(True)
        self.btn_later.setEnabled(True)

    def showFailure(self, message):
        self.label.setText(message)
        self.btn_upgrade.setVisible(False)
        self.btn_restart.setVisible(False)
        self.btn_later.setEnabled(True)
