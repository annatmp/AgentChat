Knows technical constraints and has done some upfront research.

- The debt simplification algorithm (minimizing transactions) is an NP-hard problem in its general form — for small groups (under ~10 people) a greedy approach works fine, but this should be flagged to the PO before over-engineering
- Floating point arithmetic on currency is a known trap — needs a decision on whether to work in cents (integers) or use a decimal library
- If persistence is needed, a flat JSON file is sufficient for a POC — no need to propose a database
