#!/usr/bin/env python3
"""
Sublator - Translate SRT subtitles using LLMs via OpenRouter API.

This module provides functionality to translate subtitle files in SRT format
using language models accessed through the OpenRouter API.
"""

import sys
import os
import json
import argparse
import subprocess
from urllib.request import urlopen, Request, HTTPError, URLError
from time import sleep
from typing import List, Tuple, Optional


MAX_TRANSLATE_RETRIES = 5
DEFAULT_BATCH_SIZE = 100
DEFAULT_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"


def extract_subtitles_from_video(
    video_path: str,
    track_index: int
) -> str:
    """
    Extract SRT subtitles from a video file using ffmpeg.

    Args:
        video_path: Path to the video file
        track_index: Subtitle track index to extract (0-based)

    Returns:
        SRT subtitle content as a string

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffmpeg is not available or extraction fails
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check if ffmpeg is available
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "ffmpeg is not installed or not accessible. "
            "Please install ffmpeg to use video subtitle extraction."
        ) from e

    # Extract subtitles using ffmpeg
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-map", f"0:{track_index}",
                "-f", "srt",
                "-"
            ],
            capture_output=True,
            check=True,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else "Unknown error"
        raise RuntimeError(
            f"Failed to extract subtitles from video: {error_output}"
        ) from e


def list_subtitle_streams(video_path: str) -> None:
    """
    List all available subtitle streams in a video file using ffprobe.

    Args:
        video_path: Path to the video file

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffprobe is not available or fails
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check if ffprobe is available
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            check=True,
            text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "ffprobe is not installed or not accessible. "
            "Please install ffmpeg to use video subtitle extraction."
        ) from e

    # Get stream information using ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                video_path
            ],
            capture_output=True,
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else "Unknown error"
        raise RuntimeError(
            f"Failed to probe video file: {error_output}"
        ) from e

    # Parse JSON output
    try:
        import json
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse ffprobe output: {e}"
        ) from e

    # Filter for subtitle streams
    subtitle_streams = [
        s for s in streams if s.get("codec_type") == "subtitle"
    ]

    if not subtitle_streams:
        print(f"No subtitle streams found in {video_path}", file=sys.stderr)
        sys.exit(1)

    # Print header
    print(f"Available subtitle streams in {video_path}:")

    # Print each stream
    for stream in subtitle_streams:
        index = stream.get("index", "?")
        codec = stream.get("codec_name", "unknown")
        tags = stream.get("tags", {})
        lang = tags.get("language", "unknown")
        title = tags.get("title", "")

        # Format: Stream 5: Blu-ray CEE (Russian, subrip)
        parts = [f"Stream {index}:"]
        if title:
            parts.append(title)
        parts.append(f"({lang}, {codec})")

        print(f"  {' '.join(parts)}")


def parse_srt(srt_content: str) -> List[Tuple[str, str, str]]:
    """
    Parse SRT content into a list of subtitle entries.

    Returns:
        List of tuples: (sequence_number, timestamp_line, text_content)
    """
    entries = []
    blocks = srt_content.strip().split("\n\n")

    for block in blocks:
        if not block.strip():
            continue

        lines = block.split("\n")
        if len(lines) < 3:
            continue

        sequence_number = lines[0].strip()
        timestamp_line = lines[1].strip()
        text_content = "\n".join(lines[2:])

        entries.append((sequence_number, timestamp_line, text_content))

    return entries


def format_srt(entries: List[Tuple[str, str, str]]) -> str:
    """
    Format subtitle entries back to SRT format.

    Args:
        entries: List of tuples (sequence_number, timestamp_line, text_content)

    Returns:
        SRT formatted string
    """
    if not entries:
        return ""

    result = []
    for sequence_number, timestamp_line, text_content in entries:
        result.append(sequence_number)
        result.append(timestamp_line)
        result.append(text_content)
        result.append("")  # Blank line separator

    # Join and ensure it ends with double newline
    output = "\n".join(result)
    if not output.endswith("\n\n"):
        output += "\n"

    return output


def invoke_model(model: str, prompt: str, api_key: str) -> str:
    """
    Invoke OpenRouter API to get model response.

    Args:
        model: Model identifier
        prompt: User prompt
        api_key: OpenRouter API key

    Returns:
        Model response content

    Raises:
        RuntimeError: If all retry attempts fail
    """
    req = Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }).encode("utf-8"),
    )

    for attempt in range(5):
        try:
            with urlopen(req) as res:
                response_data = json.loads(res.read().decode("utf-8"))
                return response_data["choices"][0]["message"]["content"]
        except HTTPError as e:
            msg = e.read().decode("utf-8")
            print(f"HTTP Error {e.code}: {msg}", file=sys.stderr)
        except URLError as e:
            print(f"URL Error: {e.reason}", file=sys.stderr)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to parse response: {e}", file=sys.stderr)

        if attempt < 4:
            print(f"Retrying ({attempt + 1}/5)...", file=sys.stderr)
            sleep(1.0)

    raise RuntimeError("Failed to get response from model after 5 tries.")


def parse_translation_response(
    response: str,
    expected_count: int
) -> List[Tuple[int, str]]:
    """
    Parse model response into (index, translated_text) pairs.

    Args:
        response: Model response with format "index\ntext\n---\nindex\ntext..."
        expected_count: Expected number of entries (for error messages)

    Returns:
        List of (index, translated_text) tuples

    Raises:
        ValueError: If response format is invalid
    """
    blocks = response.split("\n---\n")
    parsed_entries = []

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n", 1)  # Split into first line and rest

        if len(lines) < 1:
            raise ValueError(
                f"Entry {i+1} is empty or malformed"
            )

        index_line = lines[0].strip()
        text_content = lines[1].strip() if len(lines) > 1 else ""

        # Parse index as integer
        try:
            index = int(index_line)
        except ValueError as e:
            raise ValueError(
                f"Entry {i+1} has invalid index: '{index_line}'. "
                f"Expected a number. Error: {e}"
            ) from e

        # Validate index is positive
        if index <= 0:
            raise ValueError(
                f"Entry {i+1} has invalid index: {index}. "
                "Expected a positive integer (1, 2, 3, ...)"
            )

        parsed_entries.append((index, text_content))

    return parsed_entries


def validate_indices(
    expected_count: int,
    response_indices: List[int]
) -> Tuple[bool, str]:
    """
    Validate that response indices form a complete sequence 1..N.

    Args:
        expected_count: Expected number of entries (N)
        response_indices: List of indices from model response

    Returns:
        (is_valid, error_message) tuple
    """
    expected_set = set(range(1, expected_count + 1))
    response_set = set(response_indices)

    # Check for missing indices
    missing = expected_set - response_set
    if missing:
        return False, f"Missing indices: {sorted(missing)}"

    # Check for extra indices
    extra = response_set - expected_set
    if extra:
        return False, f"Extra indices found: {sorted(extra)}"

    # Check for duplicates in response
    if len(response_indices) != len(response_set):
        from collections import Counter
        counts = Counter(response_indices)
        duplicates = [idx for idx, count in counts.items() if count > 1]
        return False, f"Duplicate indices in response: {sorted(duplicates)}"

    return True, ""


def translate_batch(
    texts: List[str],
    target_language: str,
    model: str,
    api_key: str,
    context_entries: Optional[List[Tuple[str, str]]] = None
) -> List[str]:
    """
    Translate a batch of subtitle texts with optional context.

    Args:
        texts: List of subtitle text contents
        target_language: Target language name
        model: Model identifier
        api_key: OpenRouter API key
        context_entries: Optional list of (original, translated) tuples
                        from previous batch for context

    Returns:
        List of translated texts
    """
    # Join texts with indices and separator
    entries_with_indices = [f"{i+1}\n{text}" for i, text in enumerate(texts)]
    joined_texts = "\n---\n".join(entries_with_indices)
    expected_count = len(texts)

    # Construct prompt with optional context
    if context_entries and len(context_entries) > 0:
        # Format context entries
        context_parts = []
        for original, translated in context_entries:
            context_parts.append(f"{original}\n===\n{translated}")
        context_text = "\n---\n".join(context_parts)

        prompt = PROMPT_WITH_CONTEXT.format(
            target_language=target_language,
            previous_translations=context_text,
            batch=joined_texts,
        )
    else:
        prompt = PROMPT.format(
            target_language=target_language,
            batch=joined_texts,
        )

    for attempt in range(MAX_TRANSLATE_RETRIES):
        # Get translation
        response = invoke_model(model, prompt, api_key)

        try:
            # Parse response with indices
            parsed_response = parse_translation_response(
                response, expected_count
            )
            response_indices = [idx for idx, _ in parsed_response]

            # Validate indices
            is_valid, error_msg = validate_indices(
                expected_count, response_indices
            )

            if is_valid:
                # Sort by index to ensure correct order
                parsed_response.sort(key=lambda x: x[0])
                translations = [text for _, text in parsed_response]
                return translations

            print(
                f"Warning: Index validation failed: {error_msg}",
                file=sys.stderr
            )

        except ValueError as e:
            print(
                f"Warning: Failed to parse response: {e}",
                file=sys.stderr
            )
            error_msg = str(e)

        if attempt < MAX_TRANSLATE_RETRIES - 1:
            print(
                f"Retrying translation "
                f"(attempt {attempt + 1}/{MAX_TRANSLATE_RETRIES})...",
                file=sys.stderr
            )
            sleep(1.0)

    # All retries failed
    raise RuntimeError(
        f"Failed to translate {len(texts)} entries after "
        f"{MAX_TRANSLATE_RETRIES} attempts. "
        f"Last error: {error_msg if 'error_msg' in locals() else 'Unknown'}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles using LLMs via OpenRouter API",
        epilog=(
            "Examples:\n"
            "  cat input.srt | sublator.py --lang Spanish > output.srt\n"
            "  sublator.py --video movie.mkv --lang Spanish > output.srt"
        )
    )
    parser.add_argument(
        "-l", "--lang",
        default=None,
        help="Target language (e.g., Spanish, French, Japanese)"
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of subtitles to translate per batch "
            f"(default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help=(
            "Number of previous translations to include as context "
            "(default: batch size)"
        )
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Path to video file to extract subtitles from. "
            "If provided, stdin is ignored."
        )
    )
    parser.add_argument(
        "--stream-index",
        dest="track_index",
        type=int,
        help=(
            "Subtitle stream index to extract from video. "
            "If not provided with --video, lists available streams and exits."
        )
    )

    return parser


def main():  # pylint: disable=too-many-locals
    """Main entry point for the sublator script."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate that --stream-index requires --video
    if args.track_index is not None and args.video is None:
        print(
            "Error: --stream-index requires --video",
            file=sys.stderr
        )
        sys.exit(1)

    # If --video is provided without --stream-index, list streams and exit
    if args.video is not None and args.track_index is None:
        try:
            list_subtitle_streams(args.video)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # If --video and --stream-index are provided, --lang is required
    if args.video is not None and args.track_index is not None and args.lang is None:
        print(
            "Error: --lang is required when extracting and translating subtitles",
            file=sys.stderr
        )
        sys.exit(1)

    # For non-video mode, --lang is required
    if args.video is None and args.lang is None:
        print(
            "Error: --lang is required",
            file=sys.stderr
        )
        sys.exit(1)

    # Calculate default context size if not provided
    context_size = (
        args.context_size if args.context_size is not None
        else args.batch_size
    )

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY environment variable is not set",
            file=sys.stderr
        )
        sys.exit(1)

    # Read SRT content from video file or stdin
    if args.video:
        # Extract from video file
        try:
            print(
                f"Extracting subtitles from: {args.video} "
                f"(stream {args.track_index})",
                file=sys.stderr
            )
            srt_content = extract_subtitles_from_video(
                args.video,
                args.track_index
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin (original behavior)
        srt_content = sys.stdin.read()
        if not srt_content.strip():
            print("Error: No input provided via stdin", file=sys.stderr)
            sys.exit(1)

    # Parse SRT
    entries = parse_srt(srt_content)

    if not entries:
        print("Error: No valid subtitle entries found", file=sys.stderr)
        sys.exit(1)

    # Process in batches
    total_entries = len(entries)
    batch_size = args.batch_size
    num_batches = (total_entries + batch_size - 1) // batch_size

    translated_entries = []
    context_entries = []  # Track context for next batch

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_entries)
        batch_entries = entries[start_idx:end_idx]

        print(
            f"Translating batch {batch_idx + 1}/{num_batches} "
            f"(subtitles {start_idx + 1}-{end_idx})"
            + (f" with {len(context_entries)} context entries..."
               if context_entries else "..."),
            file=sys.stderr
        )

        # Extract texts from batch
        texts = [text for _, _, text in batch_entries]

        # Translate batch with context
        try:
            translations = translate_batch(
                texts, args.lang, args.model, api_key, context_entries
            )
        except RuntimeError as e:
            print(f"Error translating batch: {e}", file=sys.stderr)
            sys.exit(1)

        # Reconstruct entries with translations
        for i, (seq_num, timestamp, _) in enumerate(batch_entries):
            translated_entries.append((seq_num, timestamp, translations[i]))

        # Update context for next batch
        current_batch_for_context = list(zip(texts, translations))
        if len(current_batch_for_context) <= context_size:
            context_entries = current_batch_for_context
        else:
            context_entries = current_batch_for_context[-context_size:]

    # Format and output
    output = format_srt(translated_entries)
    print(output)

    print(
        f"Translation complete! Processed {total_entries} subtitles.",
        file=sys.stderr
    )

PROMPT = """
You are a professional subtitle translator. Your task is to translate movie subtitles from their original language into {target_language}.

## Instructions

1. **Translate each subtitle entry** in the batch below into {target_language}
2. **Maintain the exact format**: Each entry should have its index number on the first line and the translated text on subsequent lines
3. **Separate entries** with `---` exactly as in the input
4. **Establish consistency**: Since this is the beginning of the movie, pay special attention to how you translate character names, locations, and recurring terms, as these translations will set the standard for subsequent batches

   Example format:
   1
   Translated text here
   ---
   2
   More translated text

## Translation Guidelines

- **Natural and idiomatic**: Translate for meaning and natural flow, not word-for-word
- **Subtitle constraints**: Keep translations concise and readable within typical subtitle timing
- **Cultural adaptation**: Adapt idioms, jokes, and cultural references appropriately for the target audience
- **Character consistency**: Maintain consistent terminology for character names, locations, and recurring phrases
- **Tone and register**: Preserve the emotional tone, formality level, and speaking style of each character
- **Technical terms**: Keep proper nouns, brand names, and technical terms consistent with established conventions

## Batch to Translate

{batch}

## Your Task

Translate the above subtitle batch into {target_language} now, maintaining the exact format with index numbers and `---` separators.
"""

PROMPT_WITH_CONTEXT = """
You are a professional subtitle translator. Your task is to translate movie subtitles from their original language into {target_language}.

## Instructions

1. **Translate each subtitle entry** in the batch below into {target_language}
2. **Maintain the exact format**: Each entry should have its index number on the first line and the translated text on subsequent lines
3. **Separate entries** with `---` exactly as in the input
4. **Ensure continuity**: Use the previous translations provided as context to maintain consistency in terminology, character names, and tone throughout the movie

   Example format:
   1
   Translated text here
   ---
   2
   More translated text

## Translation Guidelines

- **Natural and idiomatic**: Translate for meaning and natural flow, not word-for-word
- **Subtitle constraints**: Keep translations concise and readable within typical subtitle timing
- **Cultural adaptation**: Adapt idioms, jokes, and cultural references appropriately for the target audience
- **Character consistency**: Maintain consistent terminology for character names, locations, and recurring phrases
- **Tone and register**: Preserve the emotional tone, formality level, and speaking style of each character
- **Technical terms**: Keep proper nouns, brand names, and technical terms consistent with established conventions

## Previous Translations (for context)

{previous_translations}

## Current Batch to Translate

{batch}

## Your Task

Translate the above subtitle batch into {target_language} now, maintaining the exact format with index numbers and `---` separators.
"""


if __name__ == "__main__":
    main()
