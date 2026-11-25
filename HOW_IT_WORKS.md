## How it works

**Forecast.** Revenue grows from the year 1 rate to the year 5 rate in even steps, and EBITDA margin moves the same way. Free cash flow = EBIT x (1 minus tax) + D&A minus capex minus the change in working capital.

**WACC.** Cost of equity uses CAPM (risk free rate + beta x equity risk premium). Debt is taxed at the company's rate. Weights use market value of equity (price x shares) and book debt.

**DCF, two ways.** Cash flows are discounted with a mid year convention. Terminal value uses either Gordon growth (final year FCF x (1 + g) / (WACC minus g)) or an exit EV/EBITDA multiple. The bridge subtracts debt and adds cash to reach equity value per share. The app also shows the exit multiple a Gordon terminal value implies, a common sanity check.

**Sensitivity.** A 5 by 5 grid shows value per share across WACC and terminal growth, shaded against today's price.

**Comps.** Peer EV/EBITDA and EV/Revenue multiples (editable) are applied to next year's forecast. The range runs from the 25th to the 75th percentile multiple.

**LBO.** A sponsor buys at an entry multiple with debt set as a multiple of EBITDA, uses levered free cash flow to repay debt each year, and exits after year 5. The app reports IRR and MOIC, an IRR grid by entry and exit multiple, and the highest price a sponsor could pay while still hitting its target IRR.

**Monte Carlo.** Thousands of simulations shift growth, margin, WACC and terminal growth at random (normal draws with the uncertainty you set) and recompute the DCF each time. The result is a distribution of value, the chance value beats today's price, and a 5th percentile downside.

**The call.** Blended value averages the two DCFs and the two comps methods. BUY needs at least 15% upside and a 60% or higher chance of beating the price; SELL needs at least 15% downside and a 40% or lower chance; anything else is HOLD.

**Limits.** Default inputs describe a fictional company. Replace them with real filings data to value a real company. This is an analysis tool, not investment advice.
