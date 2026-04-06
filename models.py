from dataclasses import dataclass


@dataclass
class Email:
    sender: str
    subject: str
    body: str
    label: str        # "work" | "personal" | "spam"
    priority: str     # "high" | "medium" | "low"
    difficulty: str = "easy"  # "easy" | "medium" | "hard"


@dataclass
class RewardBreakdown:
    classification: int = 0
    priority: int = 0
    urgency: int = 0
    archive: int = 0

    def total(self) -> int:
        return self.classification + self.priority + self.urgency + self.archive

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "priority": self.priority,
            "urgency": self.urgency,
            "archive": self.archive,
        }

    @staticmethod
    def from_dict(d: dict) -> "RewardBreakdown":
        return RewardBreakdown(
            classification=d["classification"],
            priority=d["priority"],
            urgency=d["urgency"],
            archive=d["archive"],
        )

    def pretty(self) -> str:
        return (
            f"classification: {self.classification}\n"
            f"priority: {self.priority}\n"
            f"urgency: {self.urgency}\n"
            f"archive: {self.archive}\n"
            f"total: {self.total()}"
        )

    @staticmethod
    def parse_pretty(s: str) -> "RewardBreakdown":
        data = {}
        for line in s.strip().splitlines():
            key, _, val = line.partition(":")
            key = key.strip()
            if key in ("classification", "priority", "urgency", "archive"):
                data[key] = int(val.strip())
        if len(data) != 4:
            raise ValueError(f"Malformed pretty string, got keys: {list(data.keys())}")
        return RewardBreakdown(**data)
