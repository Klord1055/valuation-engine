## How it works

**Forecast.** Revenue grows from the year 1 rate to the year 5 rate in even steps, and EBITDA margin moves the same way. Free cash flow = EBIT x (1 minus tax) + D&A minus capex minus the change in working capital.

**WACC.** Cost of equity uses CAPM (risk free rate + beta x equity risk premium). Debt is taxed at the company's rate. Weights use market value of equity (price x shares) and book debt.

**DCF, two ways.** Cash flows are discounted with a mid year convention. Long run (terminal) value uses either Gordon growth (final year FCF x (1 + g) / (WACC minus g)), discounted 4.5 years to match the mid year cash flows it is built from, or an exit EV/EBITDA multiple, discounted a full 5 years because it is a price paid at the end of year 5. The bridge subtracts debt and adds cash to reach equity value per share. The app also shows the exit multiple a Gordon terminal value implies, a common sanity check.

**Football field.** DCF bars span WACC one point higher and lower. Comps bars span the 25th to 75th percentile peer multiple. The Monte Carlo bar spans the 5th to 95th percentile.

**Sensitivity.** A 5 by 5 grid shows value per share across WACC and terminal growth, shaded against today's price.

**Comps.** Peer EV/EBITDA and EV/Revenue multiples (editable) are applied to next year's forecast. The range runs from the 25th to the 75th percentile multiple.

**LBO.** A sponsor buys at an entry multiple of next year EBITDA (the same basis as comps) with debt set as a multiple of that EBITDA. Levered free cash flow repays debt each year; once debt is gone it builds cash, and a shortfall is funded with new borrowing. Equity at exit = exit EV minus debt plus cash. The football field shows the highest price a sponsor could pay at the target IRR, with the bar spanning a required return 5 points higher and lower. If no price earns the target, the method is left out. The app reports IRR and MOIC, an IRR grid by entry and exit multiple, and the highest price a sponsor could pay while still hitting its target IRR.

**Monte Carlo.** Thousands of simulations shift growth, margin, WACC and long run growth at random (normal draws with the uncertainty you set) and recompute the blended DCF (average of the two DCF methods) each time, vectorized so 10,000 runs take milliseconds. WACC is kept at least 2% and at least 1 point above long run growth in every draw. The result is a distribution of value, the chance value beats today's price, and a 5th percentile bad case.

**The call.** Blended value averages the two DCFs and the two comps methods. BUY needs at least 15% upside and a 60% or higher chance of beating the price; SELL needs at least 15% downside and a 40% or lower chance; anything else is HOLD.

**Limits.** Default inputs describe a fictional company. Replace them with real filings data to value a real company. This is an analysis tool, not investment advice.

**What the price assumes (reverse DCF).** The app shifts every year's revenue growth by the same amount until the blended DCF value (average of the long run growth and exit multiple methods) equals today's share price, solved exactly by bisection. It does the same for EBITDA margin. The gap between your forecast and the market's implied forecast is the cushion, or the stretch, in the stock.

**What moves value the most.** Each driver moves one standard step down and up from your inputs (growth and margin ±2 points, WACC ±1 point, long run growth ±0.5 point, exit multiple ±1.0x, capex ±1 point) and the blended DCF is recomputed. Drivers are ranked by how much value swings, and the app flags any single step that pushes value across today's price.
