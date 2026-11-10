"""The parallel video writer reports progress through logging, not stdout."""

import logging
import subprocess

import pytest

from non_local_detector.visualization import _parallel_video_writer as writer

pytestmark = pytest.mark.unit


def test_ffmpeg_command_is_logged_not_printed(monkeypatch, caplog, capsys):
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or None
    )
    with caplog.at_level(logging.INFO, logger=writer.logger.name):
        writer._ffmpeg_stitch("frame_%06d.png", "out.mp4", fps=30.0)

    assert len(calls) == 1
    assert capsys.readouterr().out == ""
    assert any(
        record.name == writer.logger.name and "ffmpeg" in record.getMessage()
        for record in caplog.records
    )
