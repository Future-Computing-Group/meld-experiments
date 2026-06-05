#!/usr/bin/env python3
"""Apache Kafka transport for the SMP cross-tier deployment.

Replaces the stand-in relay (probe_broker) with the standard, published pub/sub
fabric NPS also uses (Kafka, KRaft mode). The brain and driver are unchanged in
their merge logic; only the transport differs. A unique consumer group per
subscriber gives broadcast semantics (every brain sees every delta on the topic),
and ``auto_offset_reset='earliest'`` on per-run topics removes the subscribe-before-
publish race the socket relay needed. Routing strategies (broadcast / random /
round-robin / semantic) sit ABOVE this transport, choosing the destination topic.

Sync client (kafka-python) — the brain loop is synchronous; NPS uses async aiokafka
for its broker, but the SMP brain does not need it.
"""
from __future__ import annotations

import uuid
from kafka import KafkaProducer, KafkaConsumer


class KafkaTransport:
    def __init__(self, bootstrap: str = "localhost:9092"):
        self.bootstrap = bootstrap
        self._producer = None

    def _prod(self):
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap,
                value_serializer=lambda v: v.encode("utf-8"),
                acks=1, linger_ms=5, retries=3,
                api_version_auto_timeout_ms=5000)
        return self._producer

    def publish(self, topic: str, payload: str) -> None:
        """Publish one string payload to a topic (flushed, so it is durable)."""
        self._prod().send(topic, payload)
        self._prod().flush()

    def consumer(self, topic: str, *, group: str | None = None,
                 from_beginning: bool = True, timeout_ms: int = 1000) -> KafkaConsumer:
        """A consumer on ``topic``. Default: a unique group (broadcast) reading
        from the start of the (per-run) topic, so join timing never drops a delta."""
        return KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap,
            group_id=group or ("smp-" + uuid.uuid4().hex[:10]),
            auto_offset_reset="earliest" if from_beginning else "latest",
            enable_auto_commit=True,
            value_deserializer=lambda b: b.decode("utf-8"),
            consumer_timeout_ms=timeout_ms,
            api_version_auto_timeout_ms=5000)

    def close(self):
        if self._producer is not None:
            self._producer.close(timeout=3)
            self._producer = None
