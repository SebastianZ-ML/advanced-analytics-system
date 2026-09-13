# Adaptive Multi-Agent Advanced Analytics Platform (PoC)

Functional proof-of-concept for a multi-agent architecture designed for **reproducible, defensible, and auditable business analytics**.

Converts natural language questions and disparate tables into defensible causal diagnostics, eliminating hallucinated figures through typed data contracts, independent deterministic validation, and end-to-end provenance traceability.

---

## 1. Architecture & 10-Agent Workflow

The system strictly decouples semantic reasoning and orchestration from mathematical execution and validation. Every agent produces and consumes typed artifacts adhering to **Pydantic v2** schemas:

```
[User / Business Question]
       |
       v
[Agent A: Organizer] --(ObjectiveSpec)--> Identifies decision, primary metric, periods, and ambiguities.
       |
       v
[Agent B: Data Auditor] --(DataCatalog & DataQualityReport)--> Profiles types, nulls, leading zeros, and orphans.
       |
       v
[Agent C: Methodologist] --(AnalysisPlan)--> Selects registered catalog methods with bibliographic defense.
       |
       v
[Agent D: Data Preparer] --(TransformationRecords)--> Deduplication, date cleaning, and 1.0x factor joins.
       |
       v
[Agent E: Analytical Executor] --(AnalysisResults)--> Summaries, time series, cohorts, and decompositions.
       |
       v
[Agent F: Analytical Validator] --(ValidationReport)--> Deterministic audit: 100% reconciliation, finiteness, auto-repair.
       |
       v
[Agent G: Interpreter] --(InsightReport)--> Separates proven empirical facts (result_id) from hypotheses.
       |
       v
[Agent H: Dashboard Builder] --(DashboardSpec)--> Assembles cards and charts exclusively from approved results.
       |
       v
[Agent I: Dashboard Validator] --> Verifies absence of rejected results and enforces period alignment.
       |
       v
[Agent J: Conversational Assistant] --(ChatAnswer)--> Responds with traceable citations and live filtered recalculations.
```

---

## 2. Implemented Non-Negotiable Principles

- **Immutability**: Raw files (`orders.csv`, `customers.csv`, `products.xlsx`) are read and preserved intact. Their SHA-256 hashes are calculated and stored.
- **Strict Traceability**: Every figure on the dashboard follows a verifiable audit trail:
  `Raw file -> Table profile -> Transformation -> Analytical operation -> Mathematical validation -> Visualization`.
- **Zero Quantitative Hallucination**: No LLM or agent generates arbitrary numeric figures. All calculations are executed deterministically by the DuckDB/pandas engine.
- **100% Mathematical Reconciliation**: Dimensional decompositions (waterfall) prove that `sum(dimension_deltas) == total_delta` within strict tolerance (< 0.01).
- **Join Cardinality Control**: Before joining facts and dimensions, duplicate keys are audited and the dimension table is deduplicated to guarantee an exact `1.0x` multiplication factor.
- **Preservation of Identifiers with Leading Zeros**: Columns such as `customer_id` ('00101') are audited and preserved as text to prevent loss of leading digits or truncations.
- **Explicit Handling of Incomplete Periods**: The system detects partial months (e.g., 4 days recorded in June) and raises explicit warnings to bound comparative windows to closed full months.
- **Separation of Facts and Hypotheses**: Action recommendations are tagged with `is_action_proposal_only=True` and kept strictly distinct from empirically verified findings.
- **Transparent "Demo Without LLM" Mode**: Operates completely and deterministically without requiring external API keys, identified visibly in the interface.

---

## 3. Synthetic Demonstration Dataset

Located in `data/demo/`, generated with a reproducible random seed (`seed=42`):

1. `orders.csv` (2,607 transactions):
   - Gross and net sales, units, base prices, and discounts.
   - Monthly trend with a deliberate decline concentrated in the **Wholesale / B2B** channel across April and May 2026.
   - Intentionally injected invalid dates (`2026-02-30`, `CORRUPT_DATE`).
   - Orphan order records referencing non-existent customers or products.
   - Incomplete final month (June 2026, 4 days only).
2. `customers.csv` (81 customers):
   - Text identifiers with leading zeros (`00001` through `00080`).
   - Intentional duplicate key to test deduplication (`00142`).
3. `products.xlsx`:
   - Sheet `Products`: Product catalog and base prices.
   - Sheet `Categories`: Department and category hierarchy.
4. `campaigns.csv`:
   - Marketing campaigns with budgets and active dates.
5. `ground_truth.json`:
   - Exact accounting metadata used to verify analytical accuracy.

---

## 4. Requirements & Configuration

### Prerequisites:
- Python 3.10 or higher (tested on Python 3.12).
- Modern web browser (Edge, Chrome, Firefox).

### Dependency Installation:
```bash
pip install -r backend/requirements.txt
```

### Gemini Configuration (Optional):
The platform supports two operating modes:
1. **`LLM_ENABLED`**: Utilizes Google Gemini (`gemini-3.5-flash`) via the official Google GenAI SDK for semantic structuring, method proposal, and grounded interpretations.
2. **`DEMO_WITHOUT_LLM`**: Deterministic catalog-driven mode that operates without external keys and displays the badge "[Demo: Deterministic Engine]".

To enable LLM mode, copy the example environment file and provide your API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash
GEMINI_TIMEOUT_SECONDS=60
GEMINI_MAX_RETRIES=2
```
*If `GEMINI_API_KEY` is empty or the `.env` file does not exist, the platform automatically and safely defaults to `DEMO_WITHOUT_LLM` mode.*

---

## 5. Running the Application

### 1. Start the Backend + Frontend server:
```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

### 2. Open the application:
Navigate to:
```
http://127.0.0.1:8000
```

### 3. Demonstration Walkthrough:
1. In tab **1. Projects**, click **"[Load Demo Tables]"**.
2. Click **"[Run Complete Analytical Workflow]"**.
3. Inspect the status badge in the header:
   - `[Active: Gemini gemini-3.5-flash]` or `[Demo: Deterministic Engine]`.
4. Explore each tab:
   - **2. Tables & Catalog**: Review column profiles and leading zero text preservation.
   - **3. Objective & Question**: Review Gemini assistance or deterministic rules badges.
   - **4. Quality & Relations**: Check June incomplete period warnings and orphan record handling.
   - **5. Analysis Plan**: Verify method validation against the registered catalog.
   - **6. Progress & Audit**: Observe state machine transitions and auditable transformation logs.
   - **7. Validated Dashboard**: Review executive cards, 100% reconciled waterfall charts, and engine calculation badges.
   - **8. Contextual Assistant**: Query the assistant with provenance citations, live filter recalculations, and explicit notices for missing data.
   - **9. Provenance & Audit Trail**: Inspect the step-by-step audit chain from raw files to visualizations.

---

## 6. Automated Test Suites (31 Tests Total)

The project includes two automated test suites executed with `pytest`:

### Suite A: Gemini Integration & Governance Tests (`backend/tests/test_gemini_integration.py`)
16 tests verifying decoupling, credential safety, and deterministic gatekeepers:
- Clean startup without API key and deterministic fallback operation.
- Strict Pydantic contract validation (`ObjectiveSpec`, `AnalysisPlan`, `InsightReport`).
- Methodologist Gatekeeper: strict rejection of unregistered methods or non-existent columns.
- Interpreter Gatekeeper: discard findings citing unapproved result IDs.
- Conversational assistant with audited citations and immediate detection of unanswerable queries.
- Mandatory secret redaction in logs and exception traces (`[REDACTED_API_KEY]`).
- SQLite audit logs without credential leakage.
- Resilience against prompt injection and backoff retries for 429/503 errors.
- Graceful degradation to deterministic mode on network or authentication failures.
- Exact numeric preservation matching DuckDB/pandas calculations.
- Live Gemini API call with structured JSON schema output.

```bash
python -m pytest backend/tests/test_gemini_integration.py -v
```

### Suite B: Functional Acceptance Tests (`backend/tests/test_acceptance.py`)
15 tests covering the non-negotiable acceptance criteria:

| # | Acceptance Test | Result |
|---|---|---|
| 1 | Import multiple CSVs and Excel sheets | **PASSED** |
| 2 | Preservation of identifiers with leading zeros | **PASSED** |
| 3 | Detection and prevention of joins multiplying sales | **PASSED** |
| 4 | Detection of orphan records | **PASSED** |
| 5 | Explicit handling of invalid dates | **PASSED** |
| 6 | Warning on incomplete period (June 4 days) | **PASSED** |
| 7 | Reconciliation between total variation and contributions | **PASSED** |
| 8 | Deterministic rejection of NaN or infinite values | **PASSED** |
| 9 | Propagation of rejected result to dashboard and chat | **PASSED** |
| 10 | Identical rerun with run_id versioning and idempotency | **PASSED** |
| 11 | Filters updating figures coherently | **PASSED** |
| 12 | Chatbot query triggering real analytical calculation | **PASSED** |
| 13 | Unanswerable query receives explicit limitation notice | **PASSED** |
| 14 | Empty or malformed file handling | **PASSED** |
| 15 | Cancellation or failure without deceptive state | **PASSED** |

```bash
python -m pytest backend/tests/test_acceptance.py -v
```

---

## 7. Honest Capability Statement

### Implemented End-to-End:
- Ingestion and auditing of CSV and multi-sheet `.xlsx` Excel files.
- Detection of data types, nulls, leading zeros, corrupt dates, incomplete periods, and orphan records.
- Deterministic prevention of join inflation with exact 1.0x multiplication factor checks.
- Additive dimensional decomposition of sales variance with 100% reconciliation.
- New vs recurring customer dynamics with explicit first-purchase cohort definitions.
- Independent analytical validator with automated repair loop (up to 2 attempts).
- Dashboard validator blocking the display of rejected analytical results.
- Grounded conversational assistant with 4 query classifications (explanation with provenance, live filter recalculation, new analysis detection, and explicit missing data refusal).
- SQLite persistence of projects, files, runs, audit events, and typed artifacts.
- Complete user interface built in React, TypeScript, and Tailwind CSS.

### Deliberately Out-of-Scope for Current PoC:
- **Direct SQL Connectors**: Base interface prepared; production connectors to PostgreSQL and SQL Server are slated for the next development cycle.
- **Multivariate Causal Models**: Deliberately excluded in this phase to prevent presenting unverified correlation as causal truth.
- **Advanced Forecasting Models**: Eligibility audit implemented; Prophet/ARIMA disabled when historical depth is below 12 full periods.
- **Enterprise Authentication & Role-Based Access Control (RBAC)**.
