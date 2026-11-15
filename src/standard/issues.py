"""Validation findings.

Errors break a rule of the Data Standard's section 12 (or a format the standard defines), so the
data must not be used until they are fixed. Warnings are things the standard asks for in its
prose but section 12 does not check (six decimals in label files, a final newline, unknown
fields), or that look wrong without being forbidden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.path}: {self.message} [{self.code}]"


@dataclass
class Report:
    """Everything found while validating one target (a session or a prediction run)."""

    target: str
    issues: list[Issue] = field(default_factory=list)

    def error(self, code: str, path: object, message: str) -> None:
        self.issues.append(Issue(ERROR, code, str(path), message))

    def warning(self, code: str, path: object, message: str) -> None:
        self.issues.append(Issue(WARNING, code, str(path), message))

    def add(self, severity: str, code: str, path: object, message: str) -> None:
        self.issues.append(Issue(severity, code, str(path), message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self, severity: str | None = None) -> set[str]:
        return {i.code for i in self.issues if severity is None or i.severity == severity}

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [asdict(i) for i in self.issues],
        }
