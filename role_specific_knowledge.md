# Role-specific background knowledge

This content now lives one file per role in [knowledge/](knowledge/), because each file is
hashed into the run record and loaded as that agent's private context:

| Role | File | Agent |
| ---- | ---- | ----- |
| Product Owner | [knowledge/product_owner.md](knowledge/product_owner.md) | `product_owner` |
| Backend Developer | [knowledge/backend_dev.md](knowledge/backend_dev.md) | `backend_dev` |
| Frontend Developer | [knowledge/frontend_dev.md](knowledge/frontend_dev.md) | `frontend_dev` |
| QA Engineer | [knowledge/qa_engineer.md](knowledge/qa_engineer.md) | `qa_engineer` |
| Scrum Master | [knowledge/scrum_master.md](knowledge/scrum_master.md) | `scrum_master` |
| Architect | [knowledge/architect.md](knowledge/architect.md) | `architect` |

Each agent YAML points at its own file via a `knowledge:` field, and only that agent sees it —
it is appended to that agent's system prompt, never to the shared history. Set
`role_knowledge: false` in a run config to run the same panel without any of it (factor D in
[docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md)).

Editing any of these files changes the `run_id` and invalidates previously collected results.
