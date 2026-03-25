"""Helpers for generating paper-facing artifacts from checked-in reports."""

from route_rangers.reporting.paper_artifacts import (
    build_paper_metrics,
    write_paper_artifacts,
)

__all__ = ["build_paper_metrics", "write_paper_artifacts"]
