# sublator

Translate SRT subtitle files using LLMs via OpenRouter or Z.AI API.

## Features

- Multiple API providers (OpenRouter and Z.AI)
- Batch translation for efficiency (100 subtitles per batch)
- Context-aware translation (includes previous translations for consistency)
- Support for any language
- Stdin/stdout interface for Unix pipelines
- Configurable LLM models with provider-specific defaults
- Automatic retry on API failures
- Smart validation with index-based retry (detects missing, extra, or duplicate entries)
- No external dependencies (Python standard library only)

## Requirements

- Python 3.6+
- API key for your chosen provider (OpenRouter or Z.AI)
- ffmpeg/ffprobe (required for --video option)

## Setup

Set the API key for your chosen provider:

**OpenRouter:**
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

**Z.AI:**
```bash
export ZAI_API_KEY="your-zai-api-key"
```

## Usage

```bash
cat input.srt | ./sublator.py --openrouter --lang Spanish > output.srt
```

### Provider Selection (required)

You must specify one of:
- `--openrouter`: Use OpenRouter API (requires `OPENROUTER_API_KEY`)
- `--zai`: Use Z.AI API (requires `ZAI_API_KEY`)

### Options

- `-l, --lang` (required for translation): Target language
- `-m, --model` (optional): LLM model (default: provider-specific)
  - OpenRouter: `google/gemini-2.5-flash-preview-09-2025`
  - Z.AI: `GLM-5`
- `--batch-size` (optional): Subtitles per batch (default: 100)
- `--context-size` (optional): Number of previous translations to include as context (default: batch size)
- `--video` (optional): Path to video file to extract subtitles from
- `--stream-index` (optional): Subtitle stream index to extract. If not provided with `--video`, lists available streams and exits.

### Examples

```bash
# Translate to French using OpenRouter
cat movie.srt | ./sublator.py --openrouter --lang French > movie.fr.srt

# Translate to Spanish using Z.AI
cat movie.srt | ./sublator.py --zai --lang Spanish > movie.es.srt

# Use a specific model with OpenRouter
cat show.srt | ./sublator.py --openrouter --lang Japanese --model anthropic/claude-3.5-sonnet > show.ja.srt

# Custom batch size
cat video.srt | ./sublator.py --openrouter --lang Spanish --batch-size 50 > video.es.srt

# Adjust context size for better consistency
cat series.srt | ./sublator.py --zai --lang German --context-size 75 > series.de.srt

# List available subtitle streams in a video file (no --lang needed)
./sublator.py --openrouter --video movie.mkv

# Extract and translate from video file (using stream index from listing)
./sublator.py --openrouter --video movie.mkv --stream-index 5 --lang Spanish > output.srt
```

## Testing

```bash
pytest tests.py -v
```
