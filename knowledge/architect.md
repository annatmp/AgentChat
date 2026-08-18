Knows cross-cutting structure and the constraints the wider organisation imposes. No special domain knowledge.

- The last two POCs from this team were rebuilt from scratch when they graduated to production, both times because the calculation logic was entangled with the interface; the settlement logic should sit behind a plain function boundary that any frontend can call
- There is an existing internal identity service that teams are normally expected to reuse, which conflicts directly with the "no accounts, no login" constraint — better surfaced early than discovered at review
- Anything demoed outside the team has to run from a single command; every past demo that needed infrastructure set up beforehand failed on the day
- Currency representation and rounding are cross-cutting: whichever choice is made, it has to be enforced in exactly one place, or the same rounding bugs reappear in whatever layer formats the output
- Introducing new persistent infrastructure for a POC requires an architecture review, which takes longer than this sprint
