"""Checks of the model and of the tuning, and their verdict."""

from dataclasses import dataclass

kPass = "pass"
kCaution = "caution"
kFail = "fail"
# Could not be checked: not known to be good
kUnchecked = "unchecked"

kGood = "good"
kBad = "bad"


@dataclass
class Check:
    """One check: value is what it found, criterion what it takes to pass,
    reason the finding in a few words, hint what to change when it does not
    pass and evidence the plot showing it."""

    name: str
    state: str
    value: str
    criterion: str = ""
    reason: str = ""
    hint: str = ""
    evidence: str | None = None


@dataclass
class Verdict:
    """Checks and their level: that of the worst of them."""

    checks: list

    @property
    def level(self):
        states = {check.state for check in self.checks}
        if kFail in states:
            return kBad
        if states & {kCaution, kUnchecked}:
            return kCaution
        return kGood

    @property
    def deciding(self):
        """The check that sets the level, None when it is good."""
        for state in (kFail, kCaution, kUnchecked):
            for check in self.checks:
                if check.state == state:
                    return check
        return None


def hertz(f):
    if f >= 10:
        return f"{f:.0f}"
    return f"{f:.1f}" if f >= 1 else f"{f:.2f}"


def degrees(angle):
    return f"{angle:.0f}°".replace("-", "−")
