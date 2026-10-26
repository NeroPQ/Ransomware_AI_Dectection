"""Telemetry v1: assertions are observations, never automatic malware labels."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EventSource(Record):
    kind: Literal["synthetic", "sysmon", "reviewed_import", "legacy_simulation"]
    provider: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)


class ProcessIdentity(Record):
    # A PID alone is reusable. GUID/start identity is mandatory for correlation.
    entity_id: str = Field(min_length=1)
    pid: int | None = Field(default=None, ge=0, strict=True)
    image: str | None = None
    parent_entity_id: str | None = None


class EntropyObservation(Record):
    object_type: Literal["file_content", "memory_region"]
    unit: Literal["bits_per_byte"]
    before: float | None = Field(default=None, ge=0, le=8, strict=True)
    after: float = Field(ge=0, le=8, strict=True)
    sample_bytes: int = Field(ge=256, strict=True)
    method: Literal["shannon_byte_histogram"]


class Indicator(Record):
    kind: Literal["suspected_beacon", "possible_exfiltration", "confirmed_c2", "confirmed_exfiltration",
                  "destructive_content", "ransom_note", "recovery_tamper", "injection"]
    # Reference to a separately reviewable observation, not a process-name allowlist.
    reference: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1, strict=True)


OPERATIONS = ("file_read", "file_touch", "file_create", "file_modify", "file_rename", "file_delete",
              "file_extension_change", "file_lock", "file_scramble", "ransom_note", "network_connect",
              "network_poll", "process_create", "process_access", "memory_observation", "recovery_change", "other")
Operation = Literal["file_read", "file_touch", "file_create", "file_modify", "file_rename", "file_delete",
                    "file_extension_change", "file_lock", "file_scramble", "ransom_note", "network_connect",
                    "network_poll", "process_create", "process_access", "memory_observation", "recovery_change", "other"]


class TelemetryEvent(Record):
    schema_version: Literal[1]
    source: EventSource
    host: str = Field(min_length=1)
    process: ProcessIdentity
    timestamp: datetime
    received_at: datetime | None = None
    operation: Operation
    path: str | None = None
    destination_path: str | None = None
    observation_confidence: float = Field(ge=0, le=1, strict=True)
    entropy: EntropyObservation | None = None
    indicators: list[Indicator] = Field(default_factory=list, max_length=32)
    scenario: str | None = None
    scenario_event_index: int | None = Field(default=None, ge=0)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("Schema version requires an integer")
        return value

    @field_validator("timestamp", "received_at")
    @classmethod
    def aware_timestamp(cls, value):
        if value is not None and value.utcoffset() is None:
            raise ValueError("Timezone-aware timestamps required")
        return value

    @model_validator(mode="after")
    def typed_observations(self):
        if self.operation.startswith("file_") or self.operation == "ransom_note":
            if not self.path:
                raise ValueError("File operations require path")
        if self.operation == "file_rename" and not self.destination_path:
            raise ValueError("Rename requires destination_path")
        if self.entropy:
            valid = ({"file_create", "file_modify", "file_scramble"} if self.entropy.object_type == "file_content"
                     else {"memory_observation"})
            if self.operation not in valid:
                raise ValueError("Entropy object does not match operation")
        allowed = {"suspected_beacon": {"network_connect", "network_poll"},
                   "possible_exfiltration": {"network_connect"}, "confirmed_c2": {"network_connect"},
                   "confirmed_exfiltration": {"network_connect"},
                   "destructive_content": {"file_modify", "file_scramble", "file_delete", "file_lock"},
                   "ransom_note": {"ransom_note"}, "recovery_tamper": {"recovery_change"},
                   "injection": {"process_access"}}
        if any(self.operation not in allowed[item.kind] for item in self.indicators):
            raise ValueError("Indicator does not match operation")
        return self
