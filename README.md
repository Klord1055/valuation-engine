# Valuation & Deal Engine

A Python and Streamlit app that values a company with the methods analysts already use (DCF with long run growth, DCF with an exit multiple, trading comps, and a leveraged buyout check), stress tests the DCF with Monte Carlo simulation, and turns the result into a BUY, HOLD or SELL call with a football field chart. It then works backwards: a reverse DCF shows what growth and margin today's share price already assumes, and a driver chart shows which single input could flip the call.

**Live demo:** https://klord-valuation-engine.streamlit.app

## Why I built it

My equity research work, including the DCF behind my Anterix BUY call, the Moody's credit case and the CFA Society research challenge, was all built in spreadsheets. Analysts and consultants use the same handful of methods on every company, so I turned them into one tool: change an assumption and every method, the football field, the sensitivity grid and the final call update together.

## What it does

| Tab | Output |
|---|---|
| Summary | BUY, HOLD or SELL call, blended value vs price, football field across all methods, one page CSV summary |
| What the price assumes | Reverse DCF for growth and margin, driver tornado, and which one step change pushes value across the price |
| DCF | 5 year forecast, free cash flow build, WACC build, enterprise to equity bridge, WACC by terminal growth sensitivity heatmap |
| Comps | Editable peer table, implied value from EV/EBITDA and EV/Revenue |
| LBO | Debt paydown schedule, IRR and MOIC, IRR grid by entry and exit multiple, max price at a target IRR |
| Monte Carlo | Thousands of simulated DCFs, value distribution, chance of beating the price, downside at the 5th percentile |
| Methodology | How every number is calculated |

## Tech

Python, pandas, NumPy, Plotly, Streamlit, pytest. All finance logic lives in `valuation/model.py` as pure, tested functions, separate from the interface in `app.py`.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

Default inputs describe a fictional company. Not investment advice.

Built by Klord Fernandez Cayetano.
