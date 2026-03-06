Agent-Specific Background Knowledge: Expense Splitter POC

🧑‍💼 Product Owner
Knows the business context and user research.

The primary users are groups of friends settling up after holidays or house shares — not corporate expense management
A previous attempt at this was abandoned because the settling-up math confused users; the new version must show how the result was calculated, not just the outcome
Stakeholder has explicitly said: "no accounts, no login" for this POC — it must work as a zero-signup session
There's a soft deadline because this POC will be demoed to a potential investor in the sprint review


👩‍💻 Backend Developer
Knows technical constraints and has done some upfront research.

The debt simplification algorithm (minimizing transactions) is an NP-hard problem in its general form — for small groups (under ~10 people) a greedy approach works fine, but this should be flagged to the PO before over-engineering
Floating point arithmetic on currency is a known trap — needs a decision on whether to work in cents (integers) or use a decimal library
If persistence is needed, a flat JSON file is sufficient for a POC — no need to propose a database


🎨 Frontend Developer
Knows UI/UX considerations and tooling.

A purely CLI approach will be hard to demo convincingly to a non-technical investor audience (relevant given the PO's deadline context)
A minimal web form can be built in an afternoon with no framework — but if the team decides on real-time updates (e.g. running total as you add expenses), that adds meaningful complexity
Copy/paste-friendly output (e.g. a summary you can drop into WhatsApp) came up as a desired feature in informal user conversations


🧪 QA Engineer
Knows edge cases and has seen similar tools fail.

A competing free tool (Splitwise) is well known to the target audience — QA has used it and can describe its behavior as a reference point if the team gets stuck on scope decisions
The "circular debt" case (A owes B, B owes C, C owes A) is where most implementations produce wrong or confusing output — this should be a required test case
Rounding errors across multiple people can cause the total to be off by a cent — needs an explicit decision on who absorbs the rounding remainder


🔁 Scrum Master
Knows team process context, not domain knowledge.

The team has a tendency to over-discuss the algorithm and under-specify the output format — this has caused late-stage rework before
Two team members (backend and frontend) have previously disagreed about whether the frontend should do any calculation logic or keep it strictly in the backend; this may resurface
The PO is sometimes slow to make calls under pressure — if discussions stall, try timeboxing decisions explicitly
