---
phase: 02-training-pipeline-consistency
plan: 02b
type: execute
wave: 1
depends_on: ["02-02"]
files_modified:
  - run.py
  - .paul/phases/02-training-pipeline-consistency/orchestration_audit.md
  - docs/architecture.md
autonomous: true
note: >
  Corrective maintenance fix — discovered during Plan 02-02 orchestration audit.
  Not a planned Phase 2 deliverable; does not appear in ROADMAP phase table.
  Execute between 02-02 (UNIFY) and 02-03.
---

<objective>
## Goal

Fix the broken stream confirmation subprocess path reference in `run.py` and add an explicit
`[subprocess-check]` diagnostic so path resolution is always visible in logs.

## Purpose

The orchestration audit (Plan 02-02, Task 4) found that `run.py` has three places that
reference `"stream_confirmed_movement_gate.py"` — a file that does not exist. The actual
file is `stream_confirmed_movement_gate_trade_plan_trailing.py`.

A silent fallback in `_stream_confirmed_movement_loop` prevented hard failure at runtime,
but the path resolution was never logged, making it invisible in diagnostics. This fix:

1. Corrects all three stale name references
2. Adds a `[subprocess-check]` log line that shows the resolved script path and whether it
   exists — so any future path mismatch surfaces immediately in the execution log
3. Updates documentation to reflect resolved status

No execution logic, risk controls, strategy behavior, or Alpaca configuration is changed.

## Output

- `run.py`: 4 targeted edits (1 dead constant, 1 ScannerConfig default, 1 fallback string,
  1 new diagnostic log line)
- `orchestration_audit.md`: Critical Gap entry updated to resolved
- `docs/architecture.md`: Stale filename corrected in Module Responsibilities table and DAG
</objective>

<context>
## Project Context
@.paul/PROJECT.md
@.paul/ROADMAP.md

## Audit Findings
@.paul/phases/02-training-pipeline-consistency/orchestration_audit.md

## Architecture Reference
@docs/architecture.md

## Source File
@run.py
</context>

<acceptance_criteria>

## AC-1: No remaining references to the old broken script name

```gherkin
Given: run.py after the fix
When: grep -c 'stream_confirmed_movement_gate\.py"' run.py
Then: Returns 0 — the old name appears zero times (docstrings may still reference it in text)
```

## AC-2: All three executable reference sites use the correct filename

```gherkin
Given: run.py after the fix
When: The three sites are inspected
Then:
  - STREAM_CONFIRM_SCRIPT_PATH constant resolves to stream_confirmed_movement_gate_trade_plan_trailing.py
  - ScannerConfig.stream_confirmed_movement_script default is "stream_confirmed_movement_gate_trade_plan_trailing.py"
  - The else-fallback string at the script_name assignment is "stream_confirmed_movement_gate_trade_plan_trailing.py"
```

## AC-3: [subprocess-check] diagnostic log line is emitted at path resolution time

```gherkin
Given: The stream confirmation watcher loop is entered
When: Script path is resolved (after the existing fallback block)
Then: print(f"[subprocess-check] stream_confirm_script={script_path} exists={script_path.exists()}")
      is emitted exactly once per loop start — visible in the run.py execution log
```

## AC-4: py_compile passes with no errors

```gherkin
Given: run.py after all edits
When: python -m py_compile run.py
Then: Exits 0, no SyntaxError
```

## AC-5: Docs reflect resolved status

```gherkin
Given: orchestration_audit.md and docs/architecture.md after the fix
When: The stream_confirmed entries are read
Then:
  - orchestration_audit.md Critical Gap entry is updated to "Resolved (02-02b)"
  - docs/architecture.md Module Responsibilities table uses the correct filename
  - docs/architecture.md Pipeline DAG uses the correct filename
```

</acceptance_criteria>

<tasks>

<task type="auto">
  <name>Task 1: Correct all stale path references in run.py and add [subprocess-check] diagnostic</name>
  <files>run.py</files>
  <action>
    Make four targeted edits. Read the relevant sections before editing to confirm exact context.
    Do NOT change anything else.

    ## Edit 1 — Dead constant at line ~2249 (near STREAM_CONFIRM_SCRIPT_PATH block)

    BEFORE:
    ```python
    STREAM_CONFIRM_SCRIPT_PATH = PROJECT_ROOT / "stream_confirmed_movement_gate.py"
    ```
    AFTER:
    ```python
    STREAM_CONFIRM_SCRIPT_PATH = PROJECT_ROOT / "stream_confirmed_movement_gate_trade_plan_trailing.py"
    ```
    Note: this constant is defined but never used in the launch function. Correcting it
    is hygiene — it documents the correct path. Do not wire it into the launch function.

    ## Edit 2 — ScannerConfig dataclass default at line ~2715

    BEFORE:
    ```python
    stream_confirmed_movement_script: str = "stream_confirmed_movement_gate.py"
    ```
    AFTER:
    ```python
    stream_confirmed_movement_script: str = "stream_confirmed_movement_gate_trade_plan_trailing.py"
    ```
    This is the default value used at line ~9772. Once corrected, the primary path
    resolution will point to the existing file on the first attempt.

    ## Edit 3 — Else-fallback string at line ~9772 (inside _stream_confirmed_movement_loop)

    BEFORE:
    ```python
    script_name = self.config.scanner.stream_confirmed_movement_script if self.config.scanner else "stream_confirmed_movement_gate.py"
    ```
    AFTER:
    ```python
    script_name = self.config.scanner.stream_confirmed_movement_script if self.config.scanner else "stream_confirmed_movement_gate_trade_plan_trailing.py"
    ```
    This fallback fires only when self.config.scanner is None (should not occur in production).
    Updating it closes the last stale reference.

    ## Edit 4 — Add [subprocess-check] diagnostic log after path resolution

    Locate the block in _stream_confirmed_movement_loop immediately after the existing
    alt-fallback block (lines ~9774-9777) and BEFORE the final `if not script_path.exists()`
    guard. Insert ONE print statement:

    BEFORE the final existence check:
    ```python
    if not script_path.exists():
        print(f"[stream-confirm] script missing: {script_path}")
    ```
    AFTER (insert the new line immediately above it):
    ```python
    print(f"[subprocess-check] stream_confirm_script={script_path} exists={script_path.exists()}")

    if not script_path.exists():
        print(f"[stream-confirm] script missing: {script_path}")
    ```

    This log line fires once per watcher start (inside the loop, before the restart-sleep).
    It shows the fully-resolved path (after any fallback) and whether the file exists.
    On a healthy system with a correct config it will log:
        [subprocess-check] stream_confirm_script=.../stream_confirmed_movement_gate_trade_plan_trailing.py exists=True

    ## What to avoid
    - Do NOT change the existing alt-fallback block (lines ~9774-9777) — it provides
      backward compatibility if a stale config or env still references the old name
    - Do NOT wire STREAM_CONFIRM_SCRIPT_PATH into the launch function — out of scope
    - Do NOT change the subprocess command, env dict, restart logic, or any other logic
    - Do NOT touch exit/sell logic, risk controls, or Alpaca configuration
  </action>
  <verify>
    # 1. No old name in executable positions
    grep -c 'stream_confirmed_movement_gate\.py"' run.py
    # Expected: 0

    # 2. Correct name appears in all updated sites
    grep -n 'stream_confirmed_movement_gate_trade_plan_trailing' run.py
    # Expected: 4+ lines (constant, ScannerConfig default, else-fallback, existing alt-fallback)

    # 3. Diagnostic log line present
    grep -n '\[subprocess-check\].*stream_confirm' run.py
    # Expected: 1 line

    # 4. py_compile
    python -m py_compile run.py && echo "OK"
    # Expected: OK
  </verify>
  <done>
    AC-1 satisfied: grep returns 0 for old name;
    AC-2 satisfied: all three executable sites use correct filename;
    AC-3 satisfied: [subprocess-check] log line present after alt-fallback;
    AC-4 satisfied: py_compile passes
  </done>
</task>

<task type="auto">
  <name>Task 2: Update orchestration_audit.md and docs/architecture.md</name>
  <files>
    .paul/phases/02-training-pipeline-consistency/orchestration_audit.md,
    docs/architecture.md
  </files>
  <action>
    ## orchestration_audit.md

    In Section 4 (Gap Analysis) under "Critical Gaps", find the stream_confirmed entry:

    BEFORE:
    ```
    | `stream_confirmed_movement_gate.py` does not exist | Stream confirmation subprocess
    silently fails to launch; the actual file is `stream_confirmed_movement_gate_trade_plan_trailing.py`
    | `run.py` line 2249 + 2715 | **Critical — fix before next full run** |
    ```

    AFTER: Replace the entire row with:
    ```
    | `stream_confirmed_movement_gate.py` references corrected | ✓ Resolved (Plan 02-02b):
    `STREAM_CONFIRM_SCRIPT_PATH`, `ScannerConfig.stream_confirmed_movement_script` default,
    and else-fallback string all updated to `stream_confirmed_movement_gate_trade_plan_trailing.py`.
    `[subprocess-check]` diagnostic added. Note: a silent alt-fallback already existed in
    `_stream_confirmed_movement_loop` — execution did not hard-fail, but path resolution
    was unlogged before this fix. | `run.py` (resolved) | None |
    ```

    Also, in Section 1b (Subprocess Background table), update the stream_confirmed row's
    "⚠️ Gap" note to:
    ```
    ✓ Resolved (02-02b): script path references corrected to stream_confirmed_movement_gate_trade_plan_trailing.py.
    [subprocess-check] diagnostic added.
    ```

    ## docs/architecture.md

    Make two targeted updates — filename only, no structural changes:

    **Update 1 — Module Responsibilities table**

    Find the row referencing `stream_confirmed_movement_gate.py` or the stream-confirm
    script. Update the module name column to:
    `stream_confirmed_movement_gate_trade_plan_trailing.py`

    **Update 2 — Pipeline DAG**

    In the INTRADAY LOOP section, find any reference to `stream_confirmed_movement_gate.py`
    as a PASSIVE reader. This classification was incorrect — the script IS invoked via
    subprocess (Popen background, config-gated). Update both the filename and the
    integration label:

    BEFORE (if present in PASSIVE reads block):
    ```
    stream_confirmed_movement_gate.py → reads YYYYMMDD_stream_confirmed_rebalance_orders.csv
    ```
    AFTER (move to the main Popen background section or correct in-place):
    ```
    stream_confirmed_movement_gate_trade_plan_trailing.py  [Popen background, config-gated]
          └── writes: *_stream_confirmed_rebalance_orders.csv
    ```

    Do NOT reformat or rewrite sections beyond these two filename corrections.
  </action>
  <verify>
    # orchestration_audit.md updated
    grep "Resolved.*02-02b" .paul/phases/02-training-pipeline-consistency/orchestration_audit.md
    # Expected: 1+ lines

    # docs/architecture.md updated — no old name remains
    grep "stream_confirmed_movement_gate\.py" docs/architecture.md
    # Expected: 0 results (or only in comments clearly marked as the old name)

    # docs/architecture.md has correct name
    grep "stream_confirmed_movement_gate_trade_plan_trailing" docs/architecture.md
    # Expected: 1+ results
  </verify>
  <done>
    AC-5 satisfied: orchestration_audit.md Critical Gap marked resolved; docs/architecture.md
    filename and integration classification corrected.
  </done>
</task>

</tasks>

<boundaries>

## DO NOT CHANGE

- The existing alt-fallback block in `_stream_confirmed_movement_loop` (lines ~9774–9777)
- Any subprocess command construction, env dict, or restart-on-failure logic
- Exit/sell path, risk controls, kill switches, or Alpaca configuration
- Paper/live mode safeguards (APCA_PAPER)
- Any script other than run.py in the code fix (docs only for the other two files)
- ScannerConfig fields other than `stream_confirmed_movement_script` default value
- The actual `stream_confirmed_movement_gate_trade_plan_trailing.py` script — do not rename it

## SCOPE LIMITS

- This plan corrects a stale path reference and adds a diagnostic log line only
- No new subprocess behavior — only the default path name changes
- No changes to when or how the subprocess is launched
- No changes to what data the subprocess reads or writes

</boundaries>

<verification>
Before declaring plan complete:
- [ ] `grep -c 'stream_confirmed_movement_gate\.py"' run.py` returns 0
- [ ] `grep -n 'stream_confirmed_movement_gate_trade_plan_trailing' run.py` returns 4+ lines
- [ ] `grep '\[subprocess-check\].*stream_confirm' run.py` returns 1 line
- [ ] `python -m py_compile run.py && echo OK` prints OK
- [ ] `grep "Resolved.*02-02b" .paul/phases/02-training-pipeline-consistency/orchestration_audit.md` returns a match
- [ ] `grep "stream_confirmed_movement_gate\.py" docs/architecture.md` returns 0 results
</verification>

<success_criteria>
- All four run.py edits applied with no syntax errors
- Zero remaining executable references to the old broken script name
- [subprocess-check] diagnostic visible in run.py
- Documentation reflects resolved status
- py_compile passes
</success_criteria>

<output>
After completion, create `.paul/phases/02-training-pipeline-consistency/02-02b-SUMMARY.md`
</output>
