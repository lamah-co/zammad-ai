"""Settings for Kafka connectivity and security."""

from typing import Literal

from pydantic import BaseModel, Field, FilePath, NonNegativeFloat, PositiveInt, model_validator


class KafkaSettings(BaseModel):
    """Settings related to Kafka integration."""

    silent_fallback: bool = Field(
        description="Whether to silently fallback to REST only mode if Kafka is unreachable at startup. If false, the application will fail to start if Kafka is not reachable.",
        default=False,
    )

    broker_url: str = Field(
        description="URL of the Kafka message broker notifying ticket events.",
        default="localhost:9092",
    )

    topic: str = Field(
        description="Kafka topic for ticket events",
        default="ticket-events",
    )

    client_id: str = Field(
        description="Kafka client ID used for the consumer name.",
        default="zammad-ai",
    )

    group_id: str | None = Field(
        description="Kafka consumer group ID",
        default=None,
    )

    security: "MTLSKafkaEnvSecurity | MTLSFileKafkaSecurity | DisableKafkaSecurity" = Field(
        description="Security configuration for Kafka connection.",
        default_factory=lambda: DisableKafkaSecurity(),
        discriminator="type",
    )
    event_processing: "EventProcessingSettings" = Field(
        description="Settings related to processing of incoming events.",
        default_factory=lambda: EventProcessingSettings(),
    )
    retry: "KafkaRetrySettings" = Field(
        description="Retry budget for events that fail with a retryable error.",
        default_factory=lambda: KafkaRetrySettings(),
    )
    max_poll_interval_seconds: PositiveInt = Field(
        default=900,
        description="Longest time one event may be processed before Kafka gives it to the consumer again.",
    )

    @model_validator(mode="after")
    def validate_retry_fits_poll_interval(self) -> "KafkaSettings":
        """Keep the retry waits well inside the poll interval, so a retried event is never processed twice."""
        waits: float = self.retry.total_backoff_seconds() + self.retry.max_backoff_seconds
        if waits > self.max_poll_interval_seconds / 2:
            raise ValueError(
                f"kafka.retry waits up to {waits:.0f}s per event, more than half of "
                f"kafka.max_poll_interval_seconds ({self.max_poll_interval_seconds}s). "
                "Lower the retry budget or raise the poll interval."
            )
        return self


class KafkaRetrySettings(BaseModel):
    """Retry budget and backoff for retryable event processing errors."""

    max_attempts: PositiveInt = Field(
        default=5,
        description="Attempts per event, including the first. When used up, the ticket is handed off to a human.",
    )
    initial_backoff_seconds: NonNegativeFloat = Field(
        default=2.0,
        description="Wait before the first retry. Each further retry doubles the wait.",
    )
    max_backoff_seconds: NonNegativeFloat = Field(
        default=60.0,
        description="Upper bound for one wait, also used while a hand-off cannot be written to Zammad.",
    )

    def backoff_seconds(self, attempt: int) -> float:
        """Return the wait after the given attempt: exponential, capped at max_backoff_seconds."""
        return min(self.initial_backoff_seconds * (2 ** (attempt - 1)), self.max_backoff_seconds)

    def total_backoff_seconds(self) -> float:
        """Return the sum of all waits between attempts for one event."""
        return sum(self.backoff_seconds(attempt) for attempt in range(1, self.max_attempts))


class EventProcessingSettings(BaseModel):
    """Settings related to processing of incoming events."""

    valid_request_types: list[str] = Field(
        default_factory=list,
        description="List of valid request types to process. Events with request types not in this list will be acknowledged and skipped.",
    )

    valid_action_types: list[str] = Field(
        default_factory=list,
        description="List of valid action types to process. Events with action types not in this list will be acknowledged and skipped.",
    )


class DisableKafkaSecurity(BaseModel):
    """Explicitly disable Kafka security (e.g., for plaintext connections)."""

    type: Literal["none"] = "none"


class MTLSKafkaEnvSecurity(BaseModel):
    """mTLS configuration for Kafka connection using environment variables only."""

    type: Literal["env"] = "env"

    ca_file_base64: str = Field(
        description="Base64-encoded CA certificate.",
    )

    pkcs12_base64: str = Field(
        description="Base64-encoded PKCS#12 payload.",
    )

    pkcs12_pw: str = Field(
        description="PKCS#12 password in cleartext.",
    )


class MTLSFileKafkaSecurity(BaseModel):
    """mTLS configuration for Kafka connection using file paths."""

    type: Literal["file"] = "file"

    ca_file_path: FilePath = Field(
        description="Path to the CA certificate file (PEM format).",
    )

    client_cert_path: FilePath = Field(
        description="Path to the client certificate file (PEM format).",
    )

    client_key_path: FilePath = Field(
        description="Path to the client private key file (PEM format).",
    )
