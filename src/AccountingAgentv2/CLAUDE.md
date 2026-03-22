# NMiAI Agent — CLAUDE.md

## What is this project?

This is an AI agent for **NM i AI** (Norwegian Championship in AI), a competition where an AI agent automates accounting tasks in **Tripletex** (Norwegian accounting software). The agent receives tasks in up to 7 languages (NO, NN, DE, FR, ES, PT, EN) and must complete them using the Tripletex API via a proxy.

### Scoring system
- **Correct data** earns points (each field/entity is a separate check)
- **Every write call** (POST/PUT/DELETE/PATCH) reduces score — fewer is better
- **Every 4xx error** reduces score — avoid at all costs
- **GET calls are free** — read as much as you need
- **Partial credit** — creating prerequisite entities earns points even if later steps fail
- **Speed matters** — faster completion = better score (240s timeout)

### Architecture
```
User submits task → Cloud Run endpoint (/solve)
  → classifier.py classifies task type (keyword matching, 7 languages)
  → agent.py orchestrates the loop:
      1. Prefetch reference data (departments, accounts, payment types, etc.)
      2. Try fast path for simple tasks (CREATE_EMPLOYEE, CREATE_CUSTOMER)
      3. Full agent loop: Claude API + tools until done or timeout
  → prompts.py assembles task-specific system prompt (only relevant sections)
  → tools.py has high-level tools that handle multi-step workflows
  → tripletex.py handles raw HTTP calls to Tripletex API proxy
```

### Key files
- `agent.py` — Main agent loop, prefetching, fast path logic
- `classifier.py` — Task classification via keyword matching (TaskType enum)
- `prompts.py` — System prompt sections per task type
- `tools.py` — High-level tool implementations (ToolHandler class)
- `tripletex.py` — HTTP client for Tripletex API proxy
- `analyze.py` — Log analyzer that parses TASK_REPORT entries
- `improve.sh` — Fetches Cloud Run logs → runs analyze.py → writes suggestions.md
- `deploy.sh` — Deploys to Cloud Run (europe-north1)

### Deployment
- **Cloud Run**: `nmai-agent` service in `europe-north1`, project `ai-nm26osl-1791`
- **CI/CD**: GitHub Actions on push to main (`.github/workflows/deploy.yml`)
- **Manual deploy**: `bash deploy.sh`
- **API endpoint**: `https://nmai-agent-609347898393.europe-north1.run.app/solve`

---

## How to improve the agent

### The feedback loop
1. Run `bash improve.sh` to fetch logs and analyze performance
2. Read `suggestions.md` for analysis (requires ANTHROPIC_API_KEY for AI analysis)
3. Apply improvements to the code
4. Deploy with `bash deploy.sh` or push to main

### What to optimize (priority order)

#### 1. Reduce 4xx errors (biggest score impact)
- Add validation before API calls (check required fields)
- Use pre-fetched reference data instead of guessing IDs
- Handle known Tripletex quirks in high-level tools
- Add error patterns to tools.py so they auto-recover

#### 2. Reduce write calls (POST/PUT/DELETE count)
- Use high-level tools instead of tripletex_api (they batch operations)
- Check for existing entities before creating (avoid duplicates)
- Get it right the first time — each retry is a wasted write

#### 3. Reduce iterations (faster = better)
- Extract ALL values from the prompt in one pass
- Use fast path for simple tasks (currently: CREATE_EMPLOYEE, CREATE_CUSTOMER)
- Add more task types to FAST_PATH_ELIGIBLE if they can be single-shot
- Make prompts more directive so the model doesn't explore unnecessarily

#### 4. Handle more task types correctly
- If a task type fails consistently, add specific handling in tools.py
- Add keywords to classifier.py for missed classifications
- Add prompt sections to prompts.py for edge cases

### Common failure patterns

| Pattern | Root cause | Fix location |
|---------|-----------|--------------|
| 422 on employee create | Missing department or email | `tools.py:_do_create_employee` |
| 422 on invoice GET | Missing invoiceDateFrom/To | `prompts.py` PAYMENT/INVOICE sections |
| 500 on supplier invoice | Missing currency field | `tools.py:_do_create_supplier_invoice` |
| Wrong account in voucher | Account number not in chart | `tools.py:_do_create_voucher` parent fallback |
| Timeout on corrections | Too many GET calls | `prompts.py` CORRECTIONS section (one GET rule) |
| Misclassified task | Missing keyword in classifier | `classifier.py:_KW` dict |

### Adding a new task type
1. Add to `TaskType` enum in `classifier.py`
2. Add keywords to `_KW` dict in `classifier.py`
3. Add classification logic in `classify_task()` (order matters!)
4. Add section mapping in `SECTION_MAP`
5. Add prompt section in `prompts.py:SECTIONS`
6. Optionally add a high-level tool in `tools.py`

### Adding a new high-level tool
1. Add tool definition to `TOOL_DEFINITIONS` list in `tools.py`
2. Add handler method `_do_{tool_name}` to `ToolHandler` class
3. Update `prompts.py` BASE_PROMPT TOOL SELECTION section
4. Add relevant prompt section if needed

---

## Code conventions
- Python 3.11+, async/await throughout
- Type hints on all functions
- `json.dumps(x, ensure_ascii=False)` for Norwegian characters
- Tool handlers return `dict` (auto-serialized to JSON string)
- Use `self._api()`, `self._ok()`, `self._id()`, `self._vals()` helpers
- Pre-fetched reference data via `self.ref` dict
- Logging with `logger.info/warning/error` (visible in Cloud Run logs)
- TASK_REPORT line at end of each run (parsed by analyze.py)

## Testing approach
- No unit tests currently — test by deploying and submitting tasks
- Check Cloud Run logs: `gcloud run services logs read nmai-agent --region=europe-north1 --limit=300`
- The `improve.sh` script is the primary feedback mechanism

## Important Tripletex API quirks
- GET /invoice REQUIRES `invoiceDateFrom` AND `invoiceDateTo` — omitting causes 422
- GET /supplierInvoice REQUIRES `invoiceDateFrom` AND `invoiceDateTo`
- Supplier invoice DTO: FORBIDDEN fields include `dueDate`, `amountIncludingVat`, `amount`
- Order lines: `vatType` uses `{"number": "3"}` NOT `{"id": 3}`
- Never set BOTH `unitPriceExcludingVatCurrency` AND `unitPriceIncludingVatCurrency`
- Employee creation may require department (422 if account requires it)
- `/company` endpoint may not work via proxy
- Bank account number required for invoicing (handled by `_ensure_bank_account`)
- "hourlyRates" is NOT a valid field for ProjectDTO
- `/employee/employment` endpoint — fields like employmentType do NOT exist
