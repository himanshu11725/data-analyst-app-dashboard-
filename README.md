# Personal Data Analyst

Upload a messy CSV/Excel file, get it auto-cleaned, see a dashboard (KPIs, key findings, charts),
check data health, and download the result. Free to run and free to host.

## Files
- `app.py` – Streamlit user interface
- `cleaner.py` – cleaning + profiling engine
- `insights.py` – detects Sales / Student marks / Finance data, builds KPIs, findings and charts
- `requirements.txt` – dependencies
- `sample_messy.csv`, `sample_sales.csv`, `sample_students.csv`, `sample_finance.csv` – files to try

## Run on your computer
```bash
cd data-analyst-app
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8501 and upload a sample file.

## Deploy / update on Streamlit Community Cloud
1. Upload all `.py` files and `requirements.txt` to your public GitHub repo (replace old versions).
2. The app redeploys automatically after each commit. Use Python 3.12 in Advanced settings.

## What it does
- **Cleans:** blank rows/columns, duplicates, extra spaces, placeholders (N/A, -, null), numbers stored as text
  (₹1,200, (300)), mixed date formats, inconsistent spelling/case, messy column names.
- **Dashboard:** auto-detects data type (change it in the sidebar), shows KPIs, plain-English findings and charts:
  - Sales: totals, top products/regions, monthly trend, spread, extreme values
  - Student marks: class average, pass rate, subject averages, score bands, top students, who needs support
  - Finance: money in/out, net, savings rate, spending by category, weakest month
  - Build-your-own chart: bar, line, scatter, histogram, box

## Privacy and limits
Files are processed in memory only and never saved. Limits: 20 MB, 200,000 rows.

## Next steps
3. Downloadable report  4. Login + history (Supabase)  5. Ask-in-English + AI insights
