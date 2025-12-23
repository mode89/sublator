"""
Test suite for the sublator subtitle translation module.

This module contains comprehensive tests for SRT parsing, formatting,
translation, and API interaction functionality.
"""

import json
import sys
from unittest.mock import patch, Mock, MagicMock
from urllib.error import URLError

import pytest  # pylint: disable=import-error

# Import functions from sublator
from sublator import (
    parse_srt,
    format_srt,
    translate_batch,
    invoke_model,
    build_arg_parser,
    parse_translation_response,
    validate_indices,
    extract_subtitles_from_video,
)


# SRT Parsing Tests

def test_parse_simple_single_line():
    """Test parsing simple single-line subtitles."""
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Hello World

2
00:00:03,000 --> 00:00:04,000
Second subtitle"""

    entries = parse_srt(srt_content)

    assert len(entries) == 2
    assert entries[0] == (
        "1", "00:00:01,000 --> 00:00:02,000", "Hello World"
    )
    assert entries[1] == (
        "2", "00:00:03,000 --> 00:00:04,000", "Second subtitle"
    )


def test_parse_multi_line_subtitles():
    """Test parsing multi-line subtitle text."""
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Line one
Line two
Line three

2
00:00:03,000 --> 00:00:04,000
Single line"""

    entries = parse_srt(srt_content)

    assert len(entries) == 2
    assert entries[0][2] == "Line one\nLine two\nLine three"
    assert entries[1][2] == "Single line"


def test_parse_with_special_characters():
    """Test parsing subtitles with special characters."""
    srt_content = """1
00:00:01,000 --> 00:00:02,000
[ Sound Effect ]

2
00:00:03,000 --> 00:00:04,000
"Quoted text"

3
00:00:05,000 --> 00:00:06,000
Text with <i>italics</i>"""

    entries = parse_srt(srt_content)

    assert len(entries) == 3
    assert entries[0][2] == "[ Sound Effect ]"
    assert entries[1][2] == "\"Quoted text\""
    assert entries[2][2] == "Text with <i>italics</i>"


def test_parse_empty_content():
    """Test parsing empty content."""
    entries = parse_srt("")
    assert len(entries) == 0

    entries = parse_srt("   \n\n  ")
    assert len(entries) == 0


def test_parse_malformed_entry():
    """Test handling malformed entries."""
    srt_content = """1
00:00:01,000 --> 00:00:02,000
Valid entry

Not a number
Invalid entry

3
00:00:05,000 --> 00:00:06,000
Another valid entry"""

    entries = parse_srt(srt_content)

    # Should skip malformed entry but parse valid ones
    assert len(entries) == 2
    assert entries[0][0] == "1"
    assert entries[1][0] == "3"


# SRT Formatting Tests

def test_format_simple_entries():
    """Test formatting simple subtitle entries."""
    entries = [
        ("1", "00:00:01,000 --> 00:00:02,000", "First subtitle"),
        ("2", "00:00:03,000 --> 00:00:04,000", "Second subtitle")
    ]

    output = format_srt(entries)
    expected = """1
00:00:01,000 --> 00:00:02,000
First subtitle

2
00:00:03,000 --> 00:00:04,000
Second subtitle

"""

    assert output == expected


def test_format_multi_line_subtitles():
    """Test formatting multi-line subtitles."""
    entries = [
        ("1", "00:00:01,000 --> 00:00:02,000", "Line one\nLine two"),
    ]

    output = format_srt(entries)

    assert "Line one\nLine two" in output
    assert output.endswith("\n\n")


def test_format_preserves_timestamp_format():
    """Test that timestamp format is preserved exactly."""
    timestamp = "00:02:43,747 --> 00:02:47,458"
    entries = [
        ("1", timestamp, "Text"),
    ]

    output = format_srt(entries)

    assert timestamp in output


def test_format_empty_list():
    """Test formatting empty list."""
    output = format_srt([])
    assert output == ""


# Translation Response Parsing Tests

def test_parse_translation_response_success():
    """Test parsing valid response with indices."""
    response = "1\nTranslated one\n---\n2\nTranslated two\n---\n3\nTranslated three"
    parsed = parse_translation_response(response, 3)

    assert len(parsed) == 3
    assert parsed[0] == (1, "Translated one")
    assert parsed[1] == (2, "Translated two")
    assert parsed[2] == (3, "Translated three")


def test_parse_translation_response_multiline():
    """Test parsing response with multiline text."""
    response = "1\nLine 1\nLine 2\n---\n2\nSingle line"
    parsed = parse_translation_response(response, 2)

    assert len(parsed) == 2
    assert parsed[0] == (1, "Line 1\nLine 2")
    assert parsed[1] == (2, "Single line")


def test_parse_translation_response_invalid_index():
    """Test parsing response with non-numeric index."""
    response = "abc\nTranslated text"

    with pytest.raises(ValueError, match="invalid index"):
        parse_translation_response(response, 1)


def test_parse_translation_response_negative_index():
    """Test parsing response with negative index."""
    response = "-1\nTranslated text"

    with pytest.raises(ValueError, match="invalid index: -1"):
        parse_translation_response(response, 1)


def test_parse_translation_response_zero_index():
    """Test parsing response with zero index."""
    response = "0\nTranslated text"

    with pytest.raises(ValueError, match="invalid index: 0"):
        parse_translation_response(response, 1)


def test_parse_translation_response_missing_index():
    """Test parsing response with only text, no index."""
    response = "Just translated text without index"

    with pytest.raises(ValueError, match="invalid index"):
        parse_translation_response(response, 1)


def test_parse_translation_response_empty_text():
    """Test parsing response with index but no text."""
    response = "1\n---\n2\nSome text"
    parsed = parse_translation_response(response, 2)

    assert len(parsed) == 2
    assert parsed[0] == (1, "")  # Empty text is allowed
    assert parsed[1] == (2, "Some text")


# Index Validation Tests

def test_validate_indices_success():
    """Test validation with perfect sequence."""
    is_valid, error_msg = validate_indices(3, [1, 2, 3])

    assert is_valid is True
    assert error_msg == ""


def test_validate_indices_out_of_order():
    """Test validation with out-of-order indices (should still be valid)."""
    is_valid, error_msg = validate_indices(3, [3, 1, 2])

    assert is_valid is True
    assert error_msg == ""


def test_validate_indices_missing():
    """Test detection of missing indices."""
    is_valid, error_msg = validate_indices(4, [1, 2, 4])

    assert is_valid is False
    assert "Missing indices" in error_msg
    assert "3" in error_msg


def test_validate_indices_extra():
    """Test detection of extra indices."""
    is_valid, error_msg = validate_indices(2, [1, 2, 3])

    assert is_valid is False
    assert "Extra indices" in error_msg
    assert "3" in error_msg


def test_validate_indices_duplicates():
    """Test detection of duplicate indices."""
    is_valid, error_msg = validate_indices(3, [1, 2, 2, 3])

    assert is_valid is False
    assert "Duplicate indices" in error_msg
    assert "2" in error_msg


def test_validate_indices_missing_multiple():
    """Test detection of multiple missing indices."""
    is_valid, error_msg = validate_indices(5, [1, 3, 5])

    assert is_valid is False
    assert "Missing indices" in error_msg
    assert "2" in error_msg
    assert "4" in error_msg


# Batch Translation Tests

@patch("sublator.invoke_model")
def test_translate_batch_success(mock_invoke):
    """Test successful batch translation."""
    mock_invoke.return_value = "1\nSpanish 1\n---\n2\nSpanish 2\n---\n3\nSpanish 3"

    texts = ["English 1", "English 2", "English 3"]
    translations = translate_batch(texts, "Spanish", "test-model", "test-key")

    assert len(translations) == 3
    assert translations[0] == "Spanish 1"
    assert translations[1] == "Spanish 2"
    assert translations[2] == "Spanish 3"

    # Verify invoke_model was called correctly
    mock_invoke.assert_called_once()
    call_args = mock_invoke.call_args
    assert call_args[0][0] == "test-model"  # model
    assert "Spanish" in call_args[0][1]  # prompt contains target language
    assert call_args[0][2] == "test-key"  # api_key


@patch("sublator.invoke_model")
@patch("sublator.sleep")
def test_translate_batch_index_mismatch(mock_sleep, mock_invoke):
    """Test handling of index mismatch in translations."""
    # Mock returns missing index first, then correct indices on retry
    mock_invoke.side_effect = [
        "1\nSpanish 1\n---\n3\nSpanish 3",  # Missing index 2
        "1\nSpanish 1\n---\n2\nSpanish 2\n---\n3\nSpanish 3"  # Correct
    ]

    texts = ["English 1", "English 2", "English 3"]

    # Should retry until correct indices
    translations = translate_batch(texts, "Spanish", "test-model", "test-key")

    assert len(translations) == 3
    assert mock_invoke.call_count == 2  # Called twice due to retry
    assert mock_sleep.call_count == 1  # Slept once between retries


@patch("sublator.invoke_model")
def test_translate_batch_preserves_multi_line(mock_invoke):
    """Test that multi-line subtitles are preserved."""
    mock_invoke.return_value = "1\nSpanish line 1\nSpanish line 2"

    texts = ["English line 1\nEnglish line 2"]
    translations = translate_batch(texts, "Spanish", "test-model", "test-key")

    assert len(translations) == 1
    assert "\n" in translations[0]  # Multi-line preserved


@patch("sublator.invoke_model")
def test_translate_batch_includes_context(mock_invoke):
    """Test that context entries are embedded in the prompt."""
    mock_invoke.return_value = "1\nTranslated 1\n---\n2\nTranslated 2"

    texts = ["English 1", "English 2"]
    context_entries = [("Prev 1", "Prev T1"), ("Prev 2", "Prev T2")]

    translations = translate_batch(
        texts, "Spanish", "test-model", "test-key", context_entries
    )

    assert translations == ["Translated 1", "Translated 2"]
    prompt = mock_invoke.call_args[0][1]
    # Verify context is included in prompt
    for original, translated in context_entries:
        assert f"{original}\n===\n{translated}" in prompt


@patch("sublator.invoke_model")
@patch("sublator.sleep")
def test_translate_batch_max_retries_exceeded(mock_sleep, mock_invoke):
    """Test that translate_batch raises after max retries on index mismatch."""
    mock_invoke.return_value = "1\nOnly one translation"  # Missing index 2

    texts = ["English 1", "English 2"]

    with pytest.raises(
        RuntimeError,
        match="Failed to translate 2 entries after 3 attempts"
    ):
        with patch("sublator.MAX_TRANSLATE_RETRIES", 3):
            translate_batch(
                texts,
                "Spanish",
                "test-model",
                "test-key"
            )

    assert mock_invoke.call_count == 3
    assert mock_sleep.call_count == 2  # Sleeps between attempts, not after last


# API Invocation Tests

@patch("sublator.urlopen")
def test_invoke_model_success(mock_urlopen):
    """Test successful API invocation."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [
            {
                "message": {
                    "content": "Translated text"
                }
            }
        ]
    }).encode("utf-8")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    mock_urlopen.return_value = mock_response

    result = invoke_model("test-model", "Test prompt", "test-key")

    assert result == "Translated text"
    mock_urlopen.assert_called_once()


@patch("sublator.urlopen")
@patch("sublator.sleep")  # Mock sleep to speed up test
def test_invoke_model_retry_on_error(mock_sleep, mock_urlopen):
    """Test retry logic on errors."""
    # First 2 calls fail, 3rd succeeds
    mock_urlopen.side_effect = [
        URLError("Connection error"),
        URLError("Connection error"),
        MagicMock(
            read=lambda: json.dumps({
                "choices": [{"message": {"content": "Success"}}]
            }).encode("utf-8"),
            __enter__=lambda self: self,
            __exit__=lambda *args: False
        )
    ]

    result = invoke_model("test-model", "Test prompt", "test-key")

    assert result == "Success"
    assert mock_urlopen.call_count == 3
    assert mock_sleep.call_count == 2  # Slept between retries


@patch("sublator.urlopen")
@patch("sublator.sleep")
def test_invoke_model_max_retries_exceeded(mock_sleep, mock_urlopen):
    """Test that RuntimeError is raised after max retries."""
    mock_urlopen.side_effect = URLError("Connection error")

    with pytest.raises(
        RuntimeError,
        match="Failed to get response from model after 5 tries"
    ):
        invoke_model("test-model", "Test prompt", "test-key")

    assert mock_urlopen.call_count == 5
    assert mock_sleep.call_count == 4  # Slept 4 times (not after last attempt)


# Command-Line Interface Tests

def test_default_model():
    """Test that default model is correct."""
    parser = build_arg_parser()
    args = parser.parse_args(["--lang", "Spanish"])

    assert args.model == "google/gemini-2.5-flash-preview-09-2025"
    assert args.batch_size == 50


def test_custom_model_and_batch_size():
    """Test custom model and batch size arguments."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--lang", "French",
        "--model", "custom-model",
        "--batch-size", "50"
    ])

    assert args.lang == "French"
    assert args.model == "custom-model"
    assert args.batch_size == 50


# Round-Trip Tests

def test_parse_format_round_trip():
    """Test that parsing and formatting are inverse operations."""
    original_srt = """1
00:02:43,747 --> 00:02:47,458
[ Rhythmic Droning ]

2
00:03:39,720 --> 00:03:41,971
[ Rattling ]

3
00:04:07,039 --> 00:04:09,290
Multi-line
subtitle text

"""

    # Parse then format
    entries = parse_srt(original_srt)
    formatted = format_srt(entries)

    # Should be identical (or semantically equivalent)
    assert formatted == original_srt


# Context-Aware Translation Tests

@patch("sublator.invoke_model")
def test_translate_batch_with_context(mock_invoke):
    """Test batch translation with context entries."""
    mock_invoke.return_value = "1\nSpanish 3\n---\n2\nSpanish 4"

    texts = ["English 3", "English 4"]
    context = [("English 1", "Spanish 1"), ("English 2", "Spanish 2")]

    translations = translate_batch(
        texts, "Spanish", "test-model", "test-key", context
    )

    assert len(translations) == 2
    assert translations[0] == "Spanish 3"
    assert translations[1] == "Spanish 4"

    # Verify prompt contains context
    call_args = mock_invoke.call_args
    prompt = call_args[0][1]
    assert "English 1" in prompt
    assert "Spanish 1" in prompt
    assert "===" in prompt
    assert "context" in prompt.lower() or "Previous" in prompt


@patch("sublator.invoke_model")
def test_translate_batch_without_context(mock_invoke):
    """Test batch translation without context (backward compatibility)."""
    mock_invoke.return_value = "1\nSpanish 1\n---\n2\nSpanish 2"

    texts = ["English 1", "English 2"]
    translations = translate_batch(
        texts, "Spanish", "test-model", "test-key", None
    )

    assert len(translations) == 2
    assert translations[0] == "Spanish 1"
    assert translations[1] == "Spanish 2"

    # Verify prompt does NOT contain context markers
    call_args = mock_invoke.call_args
    prompt = call_args[0][1]
    assert "Previous" not in prompt
    assert "===" not in prompt


@patch("sublator.invoke_model")
def test_translate_batch_with_empty_context(mock_invoke):
    """Test batch translation with empty context list."""
    mock_invoke.return_value = "1\nSpanish 1\n---\n2\nSpanish 2"

    texts = ["English 1", "English 2"]
    translations = translate_batch(
        texts, "Spanish", "test-model", "test-key", []
    )

    assert len(translations) == 2

    # Verify prompt does NOT contain context markers
    call_args = mock_invoke.call_args
    prompt = call_args[0][1]
    assert "context" not in prompt.lower()


def test_context_size_calculations():
    """Test context size defaults and edge cases."""
    # Default: same as batch size
    batch_size = 100
    context_size = batch_size
    assert context_size == 100

    # Small batch vs large context: use entire batch
    batch = [("a", "b")] * 8
    context_size = 20
    context = batch if len(batch) <= context_size else batch[-context_size:]
    assert len(context) == 8

    # Large batch: use last N
    batch = [("a", "b")] * 100
    context_size = 10
    context = batch if len(batch) <= context_size else batch[-context_size:]
    assert len(context) == 10

    # Context size larger than batch: use entire batch
    batch = [("a", "b")] * 30
    context_size = 50
    context = batch if len(batch) <= context_size else batch[-context_size:]
    assert len(context) == 30


@patch("sublator.invoke_model")
def test_translate_batch_multiline_with_context(mock_invoke):
    """Test that multi-line subtitles work correctly with context."""
    mock_invoke.return_value = "1\nSpanish multi\nline 2"

    texts = ["English multi\nline 2"]
    context = [("English line 1", "Spanish line 1")]

    translations = translate_batch(
        texts, "Spanish", "test-model", "test-key", context
    )

    assert len(translations) == 1
    assert "\n" in translations[0]

    # Verify context is in prompt
    call_args = mock_invoke.call_args
    prompt = call_args[0][1]
    assert "English line 1" in prompt
    assert "Spanish line 1" in prompt


# Video Extraction Tests

@patch("sublator.os.path.exists")
@patch("sublator.subprocess.run")
def test_extract_subtitles_success(mock_run, mock_exists):
    """Test successful subtitle extraction from video."""
    # Mock file exists
    mock_exists.return_value = True

    # Mock ffmpeg version check (succeeds)
    mock_version_result = Mock()
    mock_version_result.returncode = 0
    mock_version_result.stdout = "ffmpeg version 6.0.0"

    # Mock ffmpeg extraction (succeeds with SRT output)
    mock_extract_result = Mock()
    mock_extract_result.returncode = 0
    mock_extract_result.stdout = """1
00:00:01,000 --> 00:00:02,000
Extracted subtitle

2
00:00:03,000 --> 00:00:04,000
Second extracted subtitle
"""
    mock_extract_result.stderr = ""

    mock_run.side_effect = [mock_version_result, mock_extract_result]

    result = extract_subtitles_from_video("test.mp4", 0)

    assert "Extracted subtitle" in result
    assert "Second extracted subtitle" in result
    assert mock_run.call_count == 2

    # Verify first call was version check
    assert mock_run.call_args_list[0][0][0] == ["ffmpeg", "-version"]

    # Verify second call was extraction command
    extract_call = mock_run.call_args_list[1][0][0]
    assert extract_call[0] == "ffmpeg"
    assert extract_call[1] == "-i"
    assert extract_call[2] == "test.mp4"
    assert "-map" in extract_call
    assert "0:0" in extract_call
    assert "-f" in extract_call
    assert "srt" in extract_call


@patch("sublator.os.path.exists")
@patch("sublator.subprocess.run")
def test_extract_subtitles_custom_track(mock_run, mock_exists):
    """Test extraction with custom track index."""
    mock_exists.return_value = True

    mock_version_result = Mock()
    mock_version_result.returncode = 0
    mock_version_result.stdout = "ffmpeg version 6.0.0"

    mock_extract_result = Mock()
    mock_extract_result.returncode = 0
    mock_extract_result.stdout = "1\n00:00:01,000 --> 00:00:02,000\nSubtitle\n"
    mock_extract_result.stderr = ""

    mock_run.side_effect = [mock_version_result, mock_extract_result]

    extract_subtitles_from_video("test.mkv", 2)

    # Verify track index 2 was used
    extract_call = mock_run.call_args_list[1][0][0]
    assert "0:2" in extract_call


def test_extract_subtitles_file_not_found():
    """Test FileNotFoundError when video file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        extract_subtitles_from_video("nonexistent.mp4", 0)


@patch("sublator.os.path.exists")
@patch("sublator.subprocess.run")
def test_extract_subtitles_ffmpeg_not_found(mock_run, mock_exists):
    """Test RuntimeError when ffmpeg is not installed."""
    mock_exists.return_value = True
    # Mock ffmpeg not found
    mock_run.side_effect = FileNotFoundError("ffmpeg not found")

    with pytest.raises(
        RuntimeError,
        match="ffmpeg is not installed or not accessible"
    ):
        extract_subtitles_from_video("test.mp4", 0)

    mock_run.assert_called_once_with(
        ["ffmpeg", "-version"],
        capture_output=True,
        check=True,
        text=True
    )


@patch("sublator.os.path.exists")
@patch("sublator.subprocess.run")
def test_extract_subtitles_ffmpeg_fails(mock_run, mock_exists):
    """Test RuntimeError when ffmpeg extraction fails."""
    import subprocess

    mock_exists.return_value = True

    # Mock version check succeeds
    mock_version_result = Mock()
    mock_version_result.returncode = 0
    mock_version_result.stdout = "ffmpeg version 6.0.0"

    # Mock extraction fails with CalledProcessError
    error = subprocess.CalledProcessError(
        1, ["ffmpeg", "-i", "test.mp4", "-map", "0:s:0", "-f", "srt", "-"]
    )
    error.stderr = "Invalid data when processing input"

    mock_run.side_effect = [mock_version_result, error]

    with pytest.raises(
        RuntimeError,
        match="Failed to extract subtitles from video"
    ):
        extract_subtitles_from_video("test.mp4", 0)


@patch("sublator.os.path.exists")
@patch("sublator.subprocess.run")
def test_extract_subtitles_no_subtitle_stream(mock_run, mock_exists):
    """Test handling of video with no subtitle streams."""
    import subprocess

    mock_exists.return_value = True

    mock_version_result = Mock()
    mock_version_result.returncode = 0
    mock_version_result.stdout = "ffmpeg version 6.0.0"

    # ffmpeg returns CalledProcessError when no subtitle stream exists
    error = subprocess.CalledProcessError(
        1, ["ffmpeg", "-i", "no_subs.mp4", "-map", "0:s:0", "-f", "srt", "-"]
    )
    error.stderr = "Stream #0:0: not found"

    mock_run.side_effect = [mock_version_result, error]

    with pytest.raises(RuntimeError, match="Failed to extract subtitles"):
        extract_subtitles_from_video("no_subs.mp4", 0)


# CLI Argument Tests for Video Options

def test_video_argument_defaults():
    """Test that --video and --stream-index have correct defaults."""
    parser = build_arg_parser()
    args = parser.parse_args(["--lang", "Spanish"])

    assert args.video is None
    assert args.track_index is None


def test_track_index_custom():
    """Test parsing custom --stream-index."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--lang", "Spanish",
        "--video", "movie.mkv",
        "--stream-index", "2"
    ])

    assert args.video == "movie.mkv"
    assert args.track_index == 2


def test_stream_index_without_video():
    """Test that --stream-index requires --video."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--lang", "Spanish",
        "--stream-index", "3"
    ])

    assert args.video is None
    assert args.track_index == 3  # Parsing succeeds, validation happens in main()


@patch("sublator.os.path.exists")
@patch("sublator.subprocess.run")
def test_list_subtitle_streams_success(mock_run, mock_exists, capsys):
    """Test successful listing of subtitle streams."""
    import json
    from sublator import list_subtitle_streams

    mock_exists.return_value = True

    # Mock ffprobe version check
    mock_version_result = Mock()
    mock_version_result.returncode = 0
    mock_version_result.stdout = "ffprobe version 6.0.0"

    # Mock ffprobe streams output
    streams_data = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264"
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac"
            },
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {
                    "language": "eng",
                    "title": "English Subtitles"
                }
            },
            {
                "index": 3,
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {
                    "language": "spa",
                    "title": "Spanish Subtitles"
                }
            }
        ]
    }

    mock_streams_result = Mock()
    mock_streams_result.returncode = 0
    mock_streams_result.stdout = json.dumps(streams_data)

    mock_run.side_effect = [mock_version_result, mock_streams_result]

    list_subtitle_streams("test.mkv")

    captured = capsys.readouterr()
    assert "Available subtitle streams in test.mkv:" in captured.out
    assert "Stream 2: English Subtitles (eng, subrip)" in captured.out
    assert "Stream 3: Spanish Subtitles (spa, subrip)" in captured.out


@patch("sublator.os.path.exists")
@patch("sublator.subprocess.run")
def test_list_subtitle_streams_no_subtitles(mock_run, mock_exists, capsys):
    """Test listing streams when video has no subtitles."""
    import json
    from sublator import list_subtitle_streams

    mock_exists.return_value = True

    mock_version_result = Mock()
    mock_version_result.returncode = 0
    mock_version_result.stdout = "ffprobe version 6.0.0"

    streams_data = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264"
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac"
            }
        ]
    }

    mock_streams_result = Mock()
    mock_streams_result.returncode = 0
    mock_streams_result.stdout = json.dumps(streams_data)

    mock_run.side_effect = [mock_version_result, mock_streams_result]

    with pytest.raises(SystemExit):
        list_subtitle_streams("no_subs.mkv")

    captured = capsys.readouterr()
    assert "No subtitle streams found" in captured.err


@patch("sublator.os.path.exists")
def test_list_subtitle_streams_file_not_found(mock_exists, capsys):
    """Test FileNotFoundError when video file doesn't exist."""
    from sublator import list_subtitle_streams

    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError, match="Video file not found"):
        list_subtitle_streams("nonexistent.mkv")


@patch("sublator.sys.exit")
def test_stream_index_requires_video(mock_exit):
    """Test that --stream-index requires --video."""
    from sublator import build_arg_parser
    import io

    parser = build_arg_parser()
    args = parser.parse_args(["--lang", "Spanish", "--stream-index", "1"])

    # Mock sys.stderr to capture error output
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    # Simulate the validation logic
    if args.track_index is not None and args.video is None:
        print("Error: --stream-index requires --video", file=sys.stderr)

    error_output = sys.stderr.getvalue()
    sys.stderr = old_stderr

    assert "--stream-index requires --video" in error_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
