# sublator

Translate SRT subtitle files using LLMs via OpenRouter API.

## Features

- Batch translation for efficiency (50 subtitles per batch)
- Context-aware translation (includes previous translations for consistency)
- Support for any language
- Stdin/stdout interface for Unix pipelines
- Configurable LLM models
- Automatic retry on API failures
- Smart validation with index-based retry (detects missing, extra, or duplicate entries)
- No external dependencies (Python standard library only)

## Requirements

- Python 3.6+
- OpenRouter API key

## Setup

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

```bash
cat input.srt | ./sublator.py --lang Spanish > output.srt
```

### Options

- `-l, --lang` (required): Target language
- `-m, --model` (optional): LLM model (default: `google/gemini-2.5-flash-preview-09-2025`)
- `--batch-size` (optional): Subtitles per batch (default: 50)
- `--context-size` (optional): Number of previous translations to include as context (default: batch size)

### Examples

```bash
# Translate to French
cat movie.srt | ./sublator.py --lang French > movie.fr.srt

# Use a specific model
cat show.srt | ./sublator.py --lang Japanese --model anthropic/claude-3.5-sonnet > show.ja.srt

# Custom batch size
cat video.srt | ./sublator.py --lang Spanish --batch-size 50 > video.es.srt

# Adjust context size for better consistency
cat series.srt | ./sublator.py --lang German --context-size 75 > series.de.srt
```

## Testing

```bash
pytest tests.py -v
```
