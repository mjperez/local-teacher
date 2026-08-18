---
name: explain
description: "Use when the user invokes $explain to analyze a codebase, directory, file, symbol, or feature and produce a layered explanation report."
---

# Explain Skill

Dispatch 5 parallel analysis subagents - each a domain lens (structure, behavior, domain & data, external dependencies, health & risk) - against the target scope. Collect their findings and synthesize a layered report that goes from 30-second overview to deep understanding.

## Write for Humans

Everything a person reads from this skill - reports, findings, summaries, and conversation replies - MUST use plain language that a non-native English reader can follow. Before writing a report or a long reply, read `../shared/report-style.md` (resolve against the collection root (one level above this SKILL.md) - never the project's working directory) and follow it. In short: short sentences, everyday words, jargon explained on first use, the point stated first. Simplify the language, never the facts - paths, line numbers, and claims stay exact.

## Out of Scope

This skill MUST NOT:
- Modify the code being explained, even obvious typos or clearly-broken imports. The skill describes; if the user wants to fix, they invoke `$codereview` or fix manually.
- Run write-mode tools (formatter `--fix`, linter `--fix`, code generators) as part of analysis. Read-only invocations only.
- Cite benchmarks, comparisons, or "industry standards" without naming the source. Unsourced numbers read as fabrication.
- Produce value judgments in lenses 1-4 (Structure, Behavior, Domain, Dependencies). Only lens 5 (Health & Risk) assesses quality and risk. Lenses 1-4 describe and explain - they do not say "this is bad" or "this is good."
- Reproduce source code instead of pointing to it. Use `file:line_number` and short snippets (<=5 lines) for clarity. Long verbatim copies of source files are forbidden.
- Confabulate workflows that don't exist. If the scope contains only 2 confidently-identifiable workflows, trace 2 - do not invent a third to fill the count.
- Add a "fix this" or "improve this" step. `$explain` has no fix dispatch. If the user wants improvements after the report, they invoke `$codereview` or `$readability-review`.
- Skip the "uncertain" flag when the evidence is thin. Saying "unclear" or "seems like" is a valid output - guessing confidently is not.

## Invocation

Parse the user's `$explain` arguments to determine scope and options:

| Invocation | Scope |
|---|---|
| `$explain` | Whole project (default) |
| `$explain src/auth/` | Directory/module |
| `$explain src/auth/oauth.ts` | Single file |
| `$explain MyClassName` | Class/symbol (resolved via LSP or Grep) |
| `$explain "how does payment processing work?"` | Feature trace (natural language question) |
| `$explain --save` | Any of the above + write output to file |
| `$explain --save --no-html` | Save the markdown report only - skip HTML rendering for this run |
| `$explain --save --path docs/onboarding/` | Custom save location |

Arguments are combinable. Examples:
- `$explain src/auth/ --save` - explain the auth module and save the report
- `$explain --save --path docs/` - explain the whole project and save to `docs/`

If the invocation is ambiguous or unrecognizable, ask the user to clarify before proceeding.

### Scope Detection Logic

1. If the argument matches an existing file path (exact match) - **single file** mode
2. If the argument matches an existing directory (exact match) - **directory** mode
3. If no exact match, search recursively via Glob (e.g., `**/Dockerfile`, `**/MyClassName.*`) - if a unique file or directory is found, use that mode. If multiple matches, list them and ask user to clarify.
4. If no file/directory match, treat as a **symbol** name (PascalCase, camelCase, snake_case, no spaces) - resolve via LSP or Grep. If multiple matches found, list them and ask user to clarify. If zero matches, report the error.
5. If the argument contains spaces or is a question - **feature trace** mode
6. If no argument - **whole project** mode

---

## Phase 1: Preparation

### 1.0 Load User Preferences

Read `../shared/skill-context.md` for the full protocol (resolve `../shared/...` against the collection root (one level above this SKILL.md) - never the project's working directory). In brief:

1. Read `.codex/skill-context/preferences.md`.
   - If missing: invoke `$preferences` (streamlined mode - core questions only, then return here).
   - If found: load preferences.
2. Read `.codex/skill-context/explain.md` (if it exists) for explain-specific preferences.

**How preferences shape this skill:**

| Preference | Effect on Explain |
|---|---|
| Detail level: concise | Shorter TL;DR, fewer inline examples, top-level findings only |
| Detail level: detailed | Expanded sections, more code references, richer context |
| Style: analogy-heavy | Subagents use real-world parallels when describing architecture and flows |
| Style: visual-with-diagrams | More Mermaid diagrams, expanded dependency graphs, flow charts for every workflow |
| Style: example-driven | Subagents include concrete usage examples alongside descriptions |
| Assumed knowledge: beginner | Define domain terms, explain framework conventions, expand the glossary |
| Assumed knowledge: expert | Skip obvious patterns, focus on non-obvious design decisions and gotchas |
| Explain-specific: top-down | Structure agent starts with highest-level modules, drills down |
| Explain-specific: bottom-up | Structure agent starts with leaf files, builds up to the big picture |

Pass the relevant subset of preferences to each subagent prompt in Phase 2 (append after the Guiding Principles block). Subagents receive only the preferences that affect their lens - do not dump the entire file into every prompt.

### 1.1 Language Detection

Infer the primary language from file extensions in scope. If the codebase is polyglot, note all languages.

### 1.2 LSP Availability Check

Check if LSP tools are available for the detected language(s). If not, print a one-line suggestion:

> Tip: Install an LSP MCP server for [Language] for deeper analysis.

Then proceed without it. LSP is an accelerator, not a requirement.

### 1.3 File Gathering (non-project scopes only)

For **directory**, **single file**, and **symbol** scopes, resolve the file list before dispatch and pass it to all agents. This avoids 5 agents independently Globbing the same paths.

For **whole project** and **feature trace** scopes, agents explore freely - the scope is too broad to pre-load.

### 1.4 Scope Size Check

For **whole project**, **directory**, and **feature trace** scopes, estimate the number of source files in scope. If the count exceeds **500 source files**, **MUST warn** the user:

> This project has ~[N] source files. A full analysis may take several minutes. Consider narrowing scope (e.g., `$explain src/core/`). Proceed anyway? (y/n)

Proceed only after confirmation.

### 1.5 Progress Indication

Before dispatching agents, print a brief status message:

> Analyzing [scope description] across 5 lenses...

---

## Phase 2: Parallel Analysis - 5 Lenses

**MUST dispatch 5 subagents simultaneously** via the Codex agent workflow - all 5 in a single response (5 parallel Codex agent workflow calls). Sequential dispatch is a defect.

For same-platform subagents, do not pass a model or reasoning-effort override. Let every subagent inherit the parent model and reasoning effort unless the user explicitly asks for an override.

Each subagent receives a prompt containing:
- The resolved scope (what to analyze)
- The file list (for non-project scopes)
- The full Guiding Principles block (see below) - prepend it to every subagent prompt before dispatch
- The **Diagram Standards** - append the contents of `references/diagram-standards.md` (resolved against this skill's directory, not the project cwd) after the Guiding Principles, so agents that emit diagrams follow them without needing a file read
- **User preferences** (loaded in Phase 1.0) - append the relevant subset after the Guiding Principles. Format as a `## User Preferences` section with only the fields that affect that subagent's lens. For example, Agent 1 (Structure) gets explanation style and detail level; Agent 5 (Health) gets experience level and project phase.

### Guiding Principles (included in every subagent prompt)

1. **Evidence over guesswork.** Every claim about what the code does MUST reference a specific file, function, or config. No vague assertions.
2. **Flag uncertainty.** Distinguish what the code definitely does vs. what seems intended vs. what is unclear. Uncertainty is valid output - say "unclear" rather than speculate.
3. **Use LSP tools if available, fall back to Glob/Grep/Read.** Do not fail if LSP is unavailable.
4. **Respect scope boundaries.** Focus on the requested scope. Reference external context where necessary ("this module is called from `src/api/routes.ts`") but do not produce full analysis of unrelated modules.
5. **Don't reproduce source code.** Reference code with `file:line_number` pointers and short snippets for clarity. The user has the code - they need understanding, not a copy.
6. **Diagrams MUST use Mermaid syntax and follow the Diagram Standards included in this prompt** (the host appends them below - syntax patterns per diagram type, adaptive detail rules, and authoring notes). ASCII box-drawing for graphs is not permitted.
7. **No value judgments in non-Health lenses.** Agents 1-4 describe and explain. Only Agent 5 (Health & Risk) assesses quality and risk.
8. **Write for humans.** The reader may speak English as a second language. Use short sentences, everyday words, and explain jargon on first use. State the point first, then the detail. Simplify the language, never the facts - paths, line numbers, and claims stay exact.

### Subagent Roster

| # | Lens | Focus |
|---|---|---|
| 1 | Structure & Entry Points | Module boundaries, organization pattern, entry points, dependency graph, file conventions |
| 2 | Behavior - Key Workflows | Trace the 3 most important user-facing workflows end-to-end |
| 3 | Domain & Data | Data models, state transitions, domain glossary, storage mapping |
| 4 | External Dependencies | External services, infrastructure, integration patterns |
| 5 | Health & Risk | Hotspots, churn, debt, test signals, git archaeology, fastest path to competence |

### Subagent Prompt Templates

The five full prompt templates live in `references/agent-prompts.md`. Read that file, substitute the placeholders (`<GUIDING_PRINCIPLES>`, `<SCOPE_DESCRIPTION>`, and `<FILE_LIST if applicable>` - replace it with the file list for scoped runs, or delete that whole line for whole-project and feature-trace scopes that have no list), and dispatch all 5 agents in parallel.

---

## Phase 3: Synthesis

After all agents complete (or after handling failures - see Graceful Degradation), synthesize findings into the final layered report.

### Synthesis Rules

1. **TL;DR generation:** Derive from all agents - touch structure, behavior, and health. 3 sentences maximum.
2. **Orientation Cheat Sheet:** Populated by pulling specific answers from whichever agent found them. If an answer can't be determined, write "unclear" rather than guessing. Omitted for single file and symbol scopes.
3. **Deduplication:** If multiple agents mention the same file or concept, merge into the most relevant section. Don't repeat.
4. **Uncertainty preservation:** If an agent flagged something as uncertain, **MUST preserve** that flag in the synthesis. Don't silently upgrade "seems like" to "definitely."
5. **Section ordering:** Always present in the fixed order below - the layering is intentional (overview -> specifics -> actionable).
6. **Graph composition:** Combine Agent 1 and Agent 4 graphs into the top-level Architecture Overview. Deduplicate shared nodes (e.g., if both agents reference "Postgres," show it once). Internal modules at center, external services at periphery. Preserve agent-produced inline graphs in their respective sections as-is - only the top-level overview is a synthesizer composition.

### Output Structure

~~~
## TL;DR
3 sentences. What it is, why it exists, how it's organized.

## Orientation Cheat Sheet
(omitted for single file and symbol scopes)

| Question | Answer |
|---|---|
| Tech stack | ... |
| Primary language / framework | ... |
| Where are the entry points? | path/to/file |
| Where is the business logic? | path/to/dir |
| Where is the data model? | path/to/file |
| Where are the tests? | path/to/dir |
| Where is the config / env? | path/to/file |
| Who are the active contributors? | names from git |
| What's the most fragile part? | from Health agent |
| What should I not touch without care? | from Health agent |

## Architecture Overview
(omitted for single file and symbol scopes)
[Composed from Agent 1 module dependency graph + Agent 4 integration map.
Internal modules at center, external services at periphery. Deduplicate shared nodes.]

## Structure & Entry Points
[Agent 1 findings, edited for flow and deduplication - includes module dependency graph inline]

## Behavior - Key Workflows
[Agent 2 findings - the 3 traced flows with call chains and flow diagrams]

## Domain & Data
[Agent 3 findings - models, glossary, state transitions, state transition diagram if applicable]

## External Dependencies
[Agent 4 findings - includes integration map inline]

## Health & Risk
[Agent 5 findings - hotspots, debt, uncertainty]

## Fastest Path to Competence
[Extracted from Agent 5]
- Next 5 files to read
- Next 3 workflows to trace
- Areas to avoid touching without understanding first
~~~

When `--save` is not present, present the full report directly in the conversation.

---

## Phase 4: Save (optional)

Triggered only when `--save` is present.

### Default save location
`docs/explain/<scope-name>.md`

Scope name derivation:
- Whole project -> `project.md`
- Directory `src/auth/` -> `src-auth.md`
- Single file `src/auth/oauth.ts` -> `src-auth-oauth-ts.md`
- Symbol `MyClassName` -> `myclassname.md`
- Feature trace -> extract 2-4 key nouns from the question, kebab-case them (e.g., "how does payment processing work?" -> `payment-processing.md`)

### Custom save location
`--path <dir>` overrides the default directory. The filename derivation remains the same.

### Save behavior
1. Write the full synthesized report to the resolved file path. **Add YAML front-matter** at the top of the markdown with these fields:
   ```markdown
   ---
   title: "<scope-derived title>"
   generated_by: "$explain"
   generated_at: "<ISO 8601 UTC timestamp>"
   scope: "<resolved scope>"
   profile: "analytical"
   ---
   ```
2. **Render the HTML companion.** After the markdown is written, invoke the HTML renderer (best-effort; if it fails, log a warning and continue - the markdown is already saved). **Resolve `../scripts/html_render.py` to its absolute path under the collection root (one level above this SKILL.md) before running** - the command executes in the user's project cwd, which does not contain the plugin's `../scripts/` folder (invoke with `python3` where present, falling back to `python` on Windows):
   ```bash
   python ../scripts/html_render.py <report-path>.md --profile analytical
   ```
   The renderer writes `<report-path>.html` next to the markdown. On first run in a project it also creates `docs/.assets/report-lib/`. See `../shared/html-reports.md` for the protocol details.

   To skip HTML rendering for a single invocation, the user passes `--no-html`:
   ```
   $explain --save --no-html
   ```

3. Print to terminal: a brief summary (TL;DR + Cheat Sheet only) followed by the saved paths - always `Full report saved to <path>.md`, and add `and <path>.html` only if the HTML renderer succeeded. If it failed (best-effort), print the markdown path plus a one-line render warning instead.

---

## Graceful Degradation

| Situation | Behavior |
|---|---|
| 1-2 agents fail or time out | Synthesize from successful agents. Note which lenses are missing in the report. Offer to retry failed agents. |
| All agents fail | Report the failure. Suggest retrying or narrowing scope. |
| No git history available | Health & Risk agent skips git archaeology, notes "no git history available" |
| Empty/minimal repo (< 5 source files) | Skip parallel dispatch entirely. Do a single-pass direct analysis - 5 agents for a trivial codebase is wasteful. |
| Massive repo (1000+ files at scope) | Agents focus on the most significant files. Structure maps top 2 levels. Behavior traces 3 flows without exhaustive coverage. Health focuses on top hotspots. |
| Binary/generated files in scope | Skip and note their presence |
| LSP unavailable | Agents fall back to Glob/Grep/Read. No degradation in output structure, only in precision of navigation. |

---

## Error Handling

| Error | Behavior |
|---|---|
| File path argument does not exist | Report: "File not found: `<path>`. Check the path and try again." |
| Directory argument does not exist | Report: "Directory not found: `<path>`. Check the path and try again." |
| Symbol argument matches zero results | Report: "No matches found for `<symbol>`. Try a different name or use `$explain <file-path>` to target a specific file." |
| Symbol argument matches too many results (>10) | List the first 10 matches and ask user to narrow the scope. |
| Unrecognized flags | Report: "Unrecognized flag: `<flag>`. Supported flags: `--save`, `--path <dir>`." |
| `--path` points to a non-existent directory | Create the directory, or report if creation fails. |
| `--path` points to a file, not a directory | Report: "`<path>` is a file, not a directory. `--path` must point to a directory." |
| Permission error reading files | Skip unreadable files, note them in the report as "skipped (permission denied)." |
