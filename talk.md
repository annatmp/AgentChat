Multi-agent LLM systems typically fall into one of two patterns: agents take fixed turns, or a single orchestrator decides who speaks next. Both are easy to implement, but both sacrifice agent autonomy. As agents become increasingly capable and specialized, productive and optimized communication between them will be crucial to unlocking their full potential. A key part of this is deciding when it is time for an agent to speak. In this talk, we will discuss whether agents can negotiate their own speaking order, and how we might evaluate their results.

We will use a Scrum refinement meeting as our multi-agent playground. A team of role-based agents, representing classic roles from a Scrum team, must collaboratively turn a business request into a backlog of implementable user stories. We implement and compare different turn-taking strategies and evaluate these using LLM-as-a-judge across metrics including participation balance, unanswered questions, redundancy, topic drift, and urgency realism.

Attendees will learn about multi-agent communication patterns, concrete turn-taking strategies they can put into practice, comparative evaluation results, a reusable evaluation strategy, and a public repository with code and notebooks they can adapt to their own multi-agent experiments.

This idea started with a practical question: if coding agents still struggle to directly translate business requirements into working software, why not build an agentic Scrum team? The first fundamental question arose: but who speaks next? To answer this, we decided to implement and evaluate different turn-taking strategies to see which results in the most productive conversations and best overall results.

Our agentic Scrum team was tasked to refine a shared-expenses application, with various role-based agents representing classic roles from a Scrum team (Product Owner, Tech Lead, QA, Architect, etc.). They must collaboratively turn a business request into a backlog of implementable user stories under explicit rules about how turns are allocated.

After a light introduction to conversation analysis theory and game-style bidding, we present four turn-taking strategies:

Fixed-order speaking,
Importance-based self-selection where agents run a think step and emit urgency scores,
An obligation-first strategy that detects direct questions and prioritizes the addressed agent with an auction-style fallback if no questions were asked by the previous agent,
A role-based facilitator that manages an agenda and enforces constraints on speaking dominance.
We will walk through the implementation of each strategy, explaining the architecture and design decisions behind them.

Finally, we present our evaluation strategy and compare backlog quality, and conversation-level metrics. Think about participation balance, fraction of questions answered within a few turns, redundancy, topic drift, and urgency realism, all scored using LLM-as-a-judge. The implementation is lightweight, using a Python runner and notebooks, and a public repository will be shared via GitHub.

Outline (30 minutes)

0-5 min: Motivation and problem outline; limitations of naive turn-taking
5-10 min: Scenario and background; agent roles, Scrum refinement, and conversation analysis theory
10-20 min: Architecture and implemented turn-taking strategies
20-27 min: Evaluation strategy, metrics, and results
27-30 min: Learnings from building multi-agent systems, limitations
30-35 min: Open questions