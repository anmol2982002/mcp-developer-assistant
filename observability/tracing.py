"""
OpenTelemetry Tracing

Distributed tracing for request tracking.
"""

from contextlib import contextmanager
from typing import Optional

from observability.logging_config import get_logger

logger = get_logger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("opentelemetry_not_installed")


def configure_tracing(
    service_name: str = "mcp-developer-assistant",
    otlp_endpoint: Optional[str] = None,
):
    """
    Configure OpenTelemetry tracing.

    Args:
        service_name: Service name for traces
        otlp_endpoint: OTLP exporter endpoint (optional)
    """
    if not OTEL_AVAILABLE:
        logger.warning("tracing_disabled", reason="opentelemetry not installed")
        return

    try:
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        logger.info("tracing_configured", service=service_name)

    except Exception as e:
        logger.error("tracing_configuration_failed", error=str(e))


def get_tracer(name: str = "mcp"):
    """Get a tracer instance."""
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return DummyTracer()


class DummyTracer:
    """Dummy tracer when OpenTelemetry is not available."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        yield DummySpan()


class DummySpan:
    """Dummy span for no-op tracing."""

    def set_attribute(self, key, value):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass
