# sublator

Translate SRT subtitle files using LLMs via OpenRouter API.

## Features

- Batch translation for efficiency (100 subtitles per batch)
- Support for any language
- Stdin/stdout interface for Unix pipelines
- Configurable LLM models
- Automatic retry on API failures
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
- `-m, --model` (optional): LLM model (default: `google/gemini-2.5-flash-lite`)
- `--batch-size` (optional): Subtitles per batch (default: 100)

### Examples

```bash
# Translate to French
cat movie.srt | ./sublator.py --lang French > movie.fr.srt

# Use a specific model
cat show.srt | ./sublator.py --lang Japanese --model anthropic/claude-3.5-sonnet > show.ja.srt

# Custom batch size
cat video.srt | ./sublator.py --lang Spanish --batch-size 50 > video.es.srt
```

## Testing

```bash
pytest tests.py -v
```
