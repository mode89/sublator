#!/usr/bin/env python3
"""
Sublator - Translate SRT subtitles using LLMs via OpenRouter API.

This module provides functionality to translate subtitle files in SRT format
using language models accessed through the OpenRouter API.
"""

from http.client import HTTPException
import sys
import os
import json
import argparse
from urllib.request import urlopen, Request, HTTPError, URLError
from time import sleep
from typing import List, Tuple, Optional


MAX_TRANSLATE_RETRIES = 10


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
        except HTTPException as e:
            print(f"HTTP Exception: {e}\n")
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
    # Join texts with separator
    joined_texts = "\n---\n".join(texts)

    # Construct prompt with optional context
    if context_entries and len(context_entries) > 0:
        # Format context entries
        context_parts = []
        for original, translated in context_entries:
            context_parts.append(f"{original}\n===\n{translated}")
        context_text = "\n---\n".join(context_parts)

        prompt = (
            f"Here are {len(context_entries)} previous subtitles for "
            "context (DO NOT translate these - they are already "
            f"translated to {target_language} for your reference only):\n\n"
            f"{context_text}\n\n"
            f"Now translate the following {len(texts)} NEW subtitles to "
            f"{target_language}. "
            'Each subtitle is separated by "---". '
            f"You MUST output exactly {len(texts)} translations. "
            'Use "---" as separator in your response. '
            f"IMPORTANT: Output ONLY the {len(texts)} new translations below "
            "(NOT the context above):\n\n"
            f"{joined_texts}"
        )
    else:
        prompt = (
            f"Translate the following subtitles to {target_language}. "
            'Each subtitle is separated by "---". '
            "Maintain the same number of subtitles and "
            'use "---" as separator in your response. '
            "Output only the translated subtitles:\n\n"
            f"{joined_texts}"
        )

    for attempt in range(MAX_TRANSLATE_RETRIES):
        # Get translation
        response = invoke_model(model, prompt, api_key)

        # Split response by separator
        translations = response.split("\n---\n")

        # Validate count
        if len(translations) == len(texts):
            return translations

        print(
            f"Warning: Expected {len(texts)} translations "
            f"but got {len(translations)}",
            file=sys.stderr
        )
        if attempt < MAX_TRANSLATE_RETRIES - 1:
            print(
                f"Retrying translation (attempt {attempt + 1})...",
                file=sys.stderr
            )
            sleep(1.0)
        else:
            break

    raise RuntimeError(
        f"Failed to produce {len(texts)} translations after "
        f"{MAX_TRANSLATE_RETRIES} attempts."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles using LLMs via OpenRouter API",
        epilog=(
            "Example: "
            "cat input.srt | sublator.py --lang Spanish > output.srt"
        )
    )
    parser.add_argument(
        "-l", "--lang",
        required=True,
        help="Target language (e.g., Spanish, French, Japanese)"
    )
    parser.add_argument(
        "-m", "--model",
        default="google/gemini-2.5-flash-preview-09-2025",
        help=(
            "LLM model to use "
            "(default: google/gemini-2.5-flash-preview-09-2025)"
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of subtitles to translate per batch (default: 100)"
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

    return parser


def main():  # pylint: disable=too-many-locals
    """Main entry point for the sublator script."""
    parser = build_arg_parser()
    args = parser.parse_args()

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

    # Read SRT content from stdin
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


if __name__ == "__main__":
    main()
