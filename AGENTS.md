# AGENTS.md

Guidance for coding agents operating in this repository.

## Project Snapshot
- Language: Python
- Runtime shape: single-file CLI (`sublator.py`)
- Test suite: one pytest module (`tests.py`)
- Runtime dependencies: Python standard library only
- External tools at runtime: `ffmpeg`/`ffprobe` (video mode)
- Dev tooling used in repo: `pytest`, `pylint`

## Repository Layout
- `sublator.py`: production logic and CLI entrypoint
- `tests.py`: unit/behavior tests
- `.pylintrc`: lint settings (`max-line-length=80`)
- `README.md`: user-facing usage docs

## Build, Lint, and Test Commands
Use these commands as defaults.

### Syntax / Build Check
No package build step exists; use compile checks.

```bash
python3 -m py_compile sublator.py tests.py
```

### Lint
```bash
python3 -m pylint sublator.py tests.py
```

Lint notes:
- Keep lines <= 80 chars.
- Current codebase targets a clean pylint run.

### Testing

```bash
pytest tests.py
```

## Code Style Guidelines
Follow existing patterns in `sublator.py` and `tests.py`.

### Imports
- Keep imports at file top.
- Group order:
  1) standard library
  2) third-party (`pytest` in tests)
  3) local imports (`from sublator import ...`)
- Prefer explicit imports over wildcard imports.

### Formatting
- 4 spaces; no tabs.
- Maximum line length: 80.
- Prefer double-quoted strings.
- Wrap long signatures/calls vertically.
- Keep output deterministic and readable.

### Types and Signatures
- Add type hints for production functions.
- Match existing typing style (`List`, `Tuple`, `Optional`, `Callable`).
- Keep return types explicit.
- Use concise docstrings for structured tuple semantics.

### Naming
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Tests: `test_*` names with scenario + expectation
- CLI flags should map clearly to `argparse` destination names

### Docstrings and Comments
- Keep module/function docstrings concise and behavior-focused.
- Add inline comments only for non-obvious logic.
- Do not add comments that restate obvious code.

### Error Handling
- Raise specific exceptions (`FileNotFoundError`, `RuntimeError`, etc.).
- Use `raise ... from e` when wrapping lower-level errors.
- For CLI validation failures: print to `stderr`, then `sys.exit(1)`.
- For retry paths: emit actionable warnings to `stderr`.

### CLI and IO Behavior
- Preserve stdin -> stdout translation pipeline.
- Keep SRT output on stdout.
- Keep progress/warnings/errors on stderr.
- Never mix debug text into stdout subtitle output.

### API / Network Logic
- Keep provider settings centralized in `PROVIDER_CONFIGS`.
- Respect retry limits (`MAX_TRANSLATE_RETRIES`).
- Parse API responses defensively.
- Validate translated indices before accepting response content.

### Testing Patterns
- Use direct pytest assertions.
- Use `unittest.mock.patch` for side effects and external calls.
- Keep tests deterministic and isolated.
- Prefer narrow, behavior-specific test names.
- Use `capsys` for stdout/stderr assertions.

## Agent Workflow
1. Read `sublator.py` and relevant tests before editing.
2. Make the smallest coherent change.
3. Run targeted tests first.
4. Run full tests.
5. Run pylint.
6. If behavior changed, update `README.md` and tests.

## Practical Defaults
- Prefer stdlib-first implementations.
- Keep CLI flags backward compatible unless asked otherwise.
- Avoid large refactors unless clearly required.
- When uncertain, align with existing `sublator.py` patterns.
