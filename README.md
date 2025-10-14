E-commerce Growth Command Center — Quickstart & Guide
====================================================

Overview
--------
This Streamlit app ingests a single Excel workbook (uploaded at runtime) containing clickstream + transaction data and produces seven executive, funnel, retention, segmentation, product, session, and CLV modules. All computations respect a global date range filter.

Files
-----
- read.me — this quickstart and documentation (plain text)
- app.py — the Streamlit app (single file)
- requirements — pip dependency spec (no extension)

Quickstart
----------
1) Create and activate a virtual environment (example: Python 3.10+)
   - Windows (PowerShell):
       python -m venv .venv
       .\.venv\Scripts\Activate.ps1
   - macOS/Linux (bash/zsh):
       python3 -m venv .venv
       source .venv/bin/activate

2) Install dependencies:
       pip install -r requirements

3) Run the app:
       streamlit run app.py

4) In the sidebar, upload your Excel (.xlsx). If the sheet name differs from the default,
   use the sheet selector that appears after loading. Then set the date range.

Data Schema (Robust, Case-Insensitive)
--------------------------------------
The app attempts robust matching for the following fields (case-insensitive, partial match on the listed tokens).

- Timestamp (datetime): any column whose name contains one of
  [ "timestamp", "time", "date", "datetime" ]
- UserID (string): contains one of [ "userid", "user_id", "user" ]
- SessionID (string): contains one of [ "sessionid", "session_id", "session" ]
- EventType (string): contains one of [ "eventtype", "event", "action", "activity", "step" ]
- ProductID (string): contains one of [ "productid", "product_id", "sku", "item" ]
- Amount (numeric): contains one of [ "amount", "price", "revenue", "value" ]

Canonical Stage Mapping
-----------------------
A normalized event column `event_norm` is created by lower-casing the event label and mapping via regex rules:

- view|product_view|detail|browse           → view
- add|cart|atc                               → add_to_cart
- checkout|payment|pay|address|shipping      → checkout
- purchase|order|sale|success                → purchase

Guards & Transformations
------------------------
- Timestamp is coerced to datetime (invalid values dropped).
- Amount is coerced to numeric; invalid become NaN and are ignored in revenue. Negative amounts are excluded from revenue.
- All metrics aggregate only within the user-selected date range.
- If required columns are missing, the app surfaces a clear error and shows a “Column Mapping” helper to manually map fields.

Using Filters & Exports
-----------------------
- Sidebar:
  - File uploader (.xlsx), sheet selector (auto-populated), date range filter (defaults to full span).
  - "Definitions & Formulas" quick-reference (popover/expander).
- Each tab provides download buttons for table outputs (CSV).

Modules & Metrics (High-Level)
------------------------------
1) Executive Overview
   - KPIs: Total Revenue, AOV (Average Order Value), Conversion (Session→Purchase),
     Repeat Purchase Rate, Active Users (7/30-day), ARPU (Average Revenue per User).
   - Charts: Daily Revenue with 7-day moving average; New vs. Returning users over time.

2) Funnel & Drop-off (Session-level)
   - Stage conversion across {view → add_to_cart → checkout → purchase} (unique sessions).
   - Median time between stages (e.g., View→Purchase).
   - Top drop-off points by hour/day & product buckets; “stuck sessions” diagnostics.
   - Visuals: Horizontal funnel bar, time-to-purchase distribution, stuck-session table.

3) Cohorts & Retention
   - Monthly acquisition cohort = first purchase month.
   - 0–6 month retention grid (users active / cohort); cohort revenue and cumulative CLV curves.
   - Repeat interval distribution (days between purchases).

4) RFM Segmentation
   - For users with ≥1 purchase: Recency (days since last purchase), Frequency, Monetary.
   - KMeans on log-scaled R/F/M (k in [4,6], default 5). Heuristic labels: Champions, Loyal,
     Promising, At-Risk, Hibernating. Segment mix (% users/% revenue), AOV & repeat rate.

5) Product Performance & Pareto
   - Top products by revenue; Pareto curve (mark 80/20 elbow).
   - Velocity (purchases/day), Stickiness (# repeat buyers), Cannibalization watch (co-viewed but not co-purchased pairs).

6) Session Analytics
   - Per-session: events/session, duration, purchase revenue; bounce proxy (single-event, no purchase).
   - High-value patterns: most frequent event sequences among converting sessions.
   - Visuals: histograms for duration & events/session; Sankey of top flows ending in purchase.

7) CLV-Lite & Risk
   - Finite-horizon CLV (6 or 12 months). Estimate monthly spend from recent 90 days; monthly retention from recent behavior (bounded [0.2, 0.95]); discount by configurable annual rate.
   - Churn propensity (30-day) via Logistic Regression (features: recency_days, frequency, monetary, avg_ticket). Shows ROC-AUC only when labels are computable in-window.
   - Outputs a prioritized retention list (CSV export).

Notes on Methodology & Limitations
----------------------------------
- Cohorts: Acquisition cohort is defined by the month of first purchase. Retention at month `t` counts users with any event in that offset month relative to cohort start, not only purchasers.
- CLV approximation: A lightweight, finite-horizon estimate using recent spend and retention heuristics. Not a contractual forecast; sensitivity to horizon and discount rate is expected.
- Churn model: Labels require knowledge of future 30 days; if the selected date range does not include future observation windows, training metrics will be hidden.
- Seasonality: KPIs include 30/90-day and year-over-year deltas when data outside the selected window exists (internally computed safely).

Troubleshooting
---------------
- Missing columns: Use the “Column Mapping” helper if auto-detection fails.
- Non-parseable dates: Ensure your timestamp column is consistent (ISO-like preferred). Invalid rows are dropped.
- Zero purchase events: Purchase-dependent KPIs and charts are disabled with an informational note.
- Very large files: Heavy transformations are cached. If you still hit memory limits, reduce date range or pre-aggregate your data.
- Multiple sheets: The app defaults to 'ecommerce_clickstream_transacti' when present; otherwise select the correct sheet from the sidebar.
