Knows edge cases and has seen similar tools fail.

- A competing free tool (Splitwise) is well known to the target audience — QA has used it and can describe its behavior as a reference point if the team gets stuck on scope decisions
- The "circular debt" case (A owes B, B owes C, C owes A) is where most implementations produce wrong or confusing output — this should be a required test case
- Rounding errors across multiple people can cause the total to be off by a cent — needs an explicit decision on who absorbs the rounding remainder
