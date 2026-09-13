# Adaptive Multi-Agent Advanced Analytics Platform

Functional, reproducible, and verifiable multi-agent analytical platform designed to diagnose complex business problems across arbitrary tabular datasets, schemas, and domains without hardcoded assumptions or numerical hallucination.

Strictly decouples semantic interpretation and orchestration from mathematical execution and validation, enforcing an immutable single analytical source of truth per execution run.

---

## 1. Architecture & 10-Agent Workflow

Every execution run produces typed data contracts adhering to Pydantic v2 schemas. Operations are governed by an internal Execution DAG with topological sorting, cycle detection, and precondition validation:

```
[User / Business Question]
       |
       v
[Agent A: Organizer] --(ObjectiveSpec)--> Dissects question into primary metric, period, decision, and ambiguities.
       |
       v
[Agent B: Data Auditor] --(DataCatalog & DataQualityReport)--> Deterministic profiling, leading zero preservation, quarantine.
       |
       v
[Agent C: Methodologist] --(AnalysisPlan)--> Registered method selection, hypothesis formulation, DAG dependency mapping.
       |
       v
[Agent D: Data Preparer] --(TransformationRecords)--> Auditable cleaning, deduplication, 1.0x factor joins, Parquet snapshot.
       |
       v
[Agent E: Analytical Executor] --(AnalysisResults)--> Topological execution of summaries, waterfalls, cohorts, and dynamics.
       |
       v
[Agent F: Analytical Validator] --(ValidationReport)--> Independent reconciliation, finite value audit, real failure repair loop.
       |
       v
[Agent G: Interpreter] --(InsightReport)--> Grounded claims with verified numerical proof; separates facts from hypotheses.
       |
       v
[Agent H: Dashboard Builder] --(DashboardSpec)--> Assembles Apple-inspired dark mode UI exclusively from approved results.
       |
       v
[Agent I: Dashboard Validator] --> Numerical consistency verification: compares card values with calculated results.
       |
       v
[Agent J: Conversational Assistant] --(ChatAnswer)--> Grounded Q&A recalculating directly from the run Parquet snapshot.
```

---

## 2. Key Architectural Upgrades & Flaw Resolutions

The platform resolves the 8 critical diagnostic vulnerabilities identified in earlier versions:

1. **Single Analytical Source of Truth (Flaw 1 Fixed)**:
   - Every completed data preparation step persists an immutable Parquet snapshot (`SnapshotManager`) with native DuckDB I/O and SHA-256 integrity verification.
   - The entire lifecycle—analytical operations, dashboard cards, validator checks, and chat recalculations—queries this snapshot. The chat assistant never re-reads raw source files.

2. **Persistence Integrity Post-Validation (Flaw 2 Fixed)**:
   - Analysis results are validated before final database commitment and are updated with their verified statuses (`approved`, `approved_with_warnings`, or `rejected`), preventing artifacts from lingering in a misleading `pending` state.

3. **Content-Level Dashboard Card Validation (Flaw 3 Fixed)**:
   - `DashboardValidatorAgent` parses the numeric value of each metric card and cross-references it against `calculated_values` of the referenced result. Fabricated or tampered figures (e.g., $999,999,999) are rejected.

4. **Quantitative Proof for Empirical Findings (Flaw 4 Fixed)**:
   - `InterpreterAgent` requires exact numeric proof (`evidence_text` and `observed_value` present in `calculated_values`) before tagging any finding as `is_empirically_proven=True`. Speculative hypotheses remain unproven.

5. **Dynamic Schema & Column Resolution (Flaw 5 Fixed)**:
   - All engines and agents dynamically resolve date columns, metric columns, dimensions, and customer/entity identifiers via `SemanticMapper`. Operates seamlessly across retail, SaaS, healthcare, and arbitrary schemas.

6. **Real Failure-Driven Repair Loop (Flaw 6 Fixed)**:
   - When independent validation rejects an analytical result, the orchestrator triggers automated corrective actions (tightened filters, adaptive cutoff adjustment, re-execution of affected DAG subtrees) followed by real re-validation.

7. **DAG-Governed Orchestration (Flaw 7 Fixed)**:
   - Operations in `AnalysisPlan` define explicit dependencies (`depends_on`). `ExecutionDAG` performs Kahn topological sorting, cycle detection, and downstream invalidation, guaranteeing dependencies execute before dependents.

8. **Auditable Quarantine Manager (Flaw 8 Fixed)**:
   - `QuarantineManager` isolates corrupt numeric values and conflicting dimension duplicates into an auditable quarantine rather than silently coercing them to `0.0` or arbitrarily keeping the first row.

---

## 3. Multi-Domain Evaluation Datasets

The repository includes three distinct datasets exercising diverse business scenarios:

### Dataset 1: Retail & Commercial Distribution (2026 Baseline)
- **Path**: `data/demo/` (and synthetic generator `app/engine/synthetic_data.py`)
- **Tables**: `orders.csv` (2,607 transactions), `customers.csv` (81 customers), `products.xlsx` (multi-sheet), `campaigns.csv`.
- **Scenario**: Concentration of revenue contraction in the Wholesale / B2B channel between March and May 2026.
- **Edge Cases**: Corrupt dates (`2026-02-30`), orphan customer IDs, incomplete final month (June 2026, 4 days), duplicate customer keys (`00142`).

### Dataset 2: SaaS ARR & Subscription Churn (2024-2025)
- **Path**: `data/saas_subscriptions/` (generator `app/engine/multi_domain_datasets.py`)
- **Tables**: `subscriptions.csv` (179 subscription lifecycles), `accounts.csv` (121 accounts).
- **Scenario**: Monthly recurring revenue (MRR) dynamics, expansion, downgrades, and churn across plans (Starter, Growth, Professional, Enterprise).
- **Edge Cases**: Corrupt text strings in MRR, invalid start dates, conflicting account dimension duplicate.

### Dataset 3: Healthcare Clinical Encounters & Ward Operations (2025)
- **Path**: `data/healthcare_admissions/` (generator `app/engine/multi_domain_datasets.py`)
- **Tables**: `admissions.csv` (1,195 clinical encounters), `wards.csv` (6 wards), `physicians.csv` (6 physicians).
- **Scenario**: Hospital admissions, length of stay (LOS), and department cost breakdowns across Cardiology, ICU, Surgery, Oncology, Pediatrics, and Neurology.
- **Edge Cases**: Corrupt billing strings (`UNKNOWN_BILLING_ERR`), invalid calendar day (`2025-02-31`).

---

## 4. Design & User Interface

The web interface features an **Apple macOS-inspired dark premium visual system**:
- **Palette**: Deep graphite base (`#0A0A0B`), secondary surfaces (`#121214`, `#18181B`), and restrained Apple Pro Blue (`#2997FF`) accents.
- **Surfaces**: Subtle Liquid Glass effects with `backdrop-filter: blur(24px)`, diffuse drop shadows, and 1px border highlights.
- **Visualizations**: Dynamic Chart.js dark-mode waterfall breakdowns (positive contributions in blue, negative in soft red `#FF453A`) and smoothed Bezier monthly trends.
- **Auditability**: Complete underlying data tables and full provenance chains visible below every chart.
- **Compliance**: Zero emojis and 100% English language throughout the codebase, documentation, tests, and user interface.

---

## 5. Installation & Execution

### Prerequisites:
- Python 3.10+ (tested on Python 3.12).
- Modern web browser (Chrome, Edge, Firefox).

### 1. Install Dependencies:
```bash
pip install -r backend/requirements.txt
```

### 2. Configure Environment (Optional):
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
- **Mode 1 (`LLM_ENABLED`)**: Set `GEMINI_API_KEY=your_key`. Uses Gemini (`gemini-3.5-flash`) for semantic structuring, method suggestions, and narrative insights.
- **Mode 2 (`DEMO_WITHOUT_LLM`)**: Leave `GEMINI_API_KEY` blank or unset. The platform executes entirely with the deterministic DuckDB/pandas engine.

### 3. Launch Application:
```bash
# From repository root
python run.py
```
Or start backend and frontend independently:
```bash
# Backend (FastAPI on port 8000)
cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Frontend
Open frontend/index.html in any modern browser
```

---

## 6. Verification & Automated Test Suites

The project includes 5 comprehensive test suites covering 51 automated tests:

| Test Suite | Purpose | Tests |
| :--- | :--- | :---: |
| `tests/test_acceptance.py` | 15 non-negotiable acceptance criteria (joins, zero preservation, reconciliations, idempotency, etc.) | 15 |
| `tests/test_adaptable_platform.py` | Multi-domain validation across SaaS, Healthcare, and Retail datasets | 10 |
| `tests/test_flaws_reproduction.py` | Regression guards proving that all 8 previous architectural flaws are fixed | 6 |
| `tests/test_gemini_integration.py` | LLM decoupling, prompt injection defense, timeout/retry, quota fallback, deterministic preservation | 16 |
| `tests/test_stage1_verification.py` | Parquet snapshots, SHA-256 integrity, approved status persistence, grounded chat recalculation | 4 |

### Run Complete Test Suite:
```bash
cd backend
python -m pytest tests/ -v
```