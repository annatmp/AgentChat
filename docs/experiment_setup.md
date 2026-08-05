# Experimentation Guide

This Guide describes the general experimental setup to answer the following research questions:

(1) Do agent scrum teams create better plans if they work as a team?
(2) Are homogeneous team more effective (i.e. all members of the team are an instance of the same model)
(3) Which turn-taking mechanism leads to the best results?
(4) Behavioural questions about agents in teams:
(4a) How eager are agents? Do they consistently bid the highest in bidding style turn taking?
(4b) In mixed teams, do models prefer to give the word more often to another instantiation of their own model?
(4c) Are there differences in the models in selective turn taking strategies? 

## Setup:

3 different setups:

1. One agent "team"
2. Homgeneous teams
3. Mixed teams


### One agent "team"

This is the base-case, which is what happens now mostly. One agent is queried to device the plan. It will get all the information the different roles are also getting and it is asked to review the plan under the lense of all those different positions. This makes 3 teams for 3 model families.


### Homogeneous teams

These are actual teams with the 6 different roles. Each agent instance receives the information it gets from its role. Each agent is instantiated with the same model. Assuming 3 model families, we get 3 different teams.

### Mixed Teams

These are teams like the homgeneous teams, but with mixed model families. To avoid having to serve all permutation, we split the team in 3 pairs and assign the three model families to each pair once. That makes 3 different mixed teams.

## Runs

Each team runs 5 times per strategy. 

| Setup  | Number Teams  | Number Strategy | Repitition | Total Runs |
|---|---|---|---|---|
| One agent | 3 | 1\* | 5 | 15 |
| Homogeneous | 3 | 4 | 5 | 60 |
| Mixed | 3 | 4 | 5 | 60 |
|**TOTAL**| | | |135|


\* One agent teams do not need turn taking strategies

## Turns

Every run will receive the same number of turns. This will be the total number of messages send (not per agent), as different strategies may create imbalances in how often an agent speaks. One agent runs will also receive the same number of turns.

## Early Stopping

All runs will have the option of early stopping. Before every new turn, agents will get the option to vote if they think this conversation has converged or not. They are also allowed to openly discuss stopping this conversation earlier, by e.g. inviting agents to vote "close" on the next converstion. They are all fully aware of the mechanism. 
One agent conversations can also terminate if they feel like they reached a good state.


## Summarizer

At the end of each run, the conversation is summarized by a "summarizer" to make sure all input follows the same standards and that conversation between agents that do not reach convergence still give output.

The summarizer reviews the conversation and makes the final deliverable that will be used for evaluation later.

To keep this controlled, the same model will be used as summarizer for every setting. The model that is used for summarizing is itself not participating in any teams nor will it be present in the LLM-as-a-Judge panel. 

*Motativation* LLM-as-a-judge research suggests that models rate output from models in their own family more favourably. We will set up a diverse panel of judges to prevent this, but by also letting an independent model create the plan, no. model will directly rate the output of their own family. It will always be filtered through at least one independent model

## LLM-as-a-judge

### Panel

All models that are part of the teams + one independentm model family (that is also not the same family as the summarizer) will be used for the judge panel. Each judge rates a plan 5 times. All 20 scores for a run will be stored and assessed. We will look specifically at agreement and std to also rate our evaluation metrics.

### Verification

We will further select X number of rated project proposals and hand them to human annotators. They will receive the same instructions than the LLM's but are merely asked to rang the plans. We will investigate if their ranking aligns with the ranking of the LLMs. To make this more feasible, we will pick only plans that differ significantly in scores and should therefore be easy to rank. 

Further, if we notice that certain plans get ratings with a high std, we might also hand them over to a human evaluator.

## Model Selection

We will select mid-tier models of the following providers:

Team and Judges:
- GPT (GPT-5.6 Terra)
- Anthropic (Sonnet 5)
- Google (gemini-3.6-flash)

Independent Judge
- Mistral Medium mistral-medium-2505

Summarizer:
- DeepSeak

## Limitations

Limitations of this project are the followoing:

(1) We are limited by (a) the amount of models we include in this story an (b) will not test all possible permutations
(2) Even though we use a component of human evaluation, we base our result on LLM-as-a-judge
(3) Measuring the quality of a implementation plan is not a purely objective task and we expect a not perfect inter-annotator-agreeement
(4) We focus on turn-taking mostly and team composition second. Much more research will need to be done in conversation design, prompting, role-creation etc. 


## Pricing

**Google**

| |per 1M tokens in USD|
|---|---
Input price	|$1.50
Output price (including thinking tokens)|$7.50

**Anthropic**

Model pricing
The following table shows pricing for all Claude models:

|Model |	Base Input Tokens |	5m Cache Writes |1h Cache Writes|	Cache Hits & Refreshes	|Output Tokens
|---|---|---|---|---|---|
Claude Sonnet 5  through August 31, 2026	|$2 / MTok	|$2.50 / MTok	|$4 / MTok	|$0.20 / MTok	|$10 / MTok
Claude Sonnet 5starting September 1, 2026|	$3 / MTok	|$3.75 / MTok	|$6 / MTok	|$0.30 / MTok|	$15 / MTok|


GPT-5.6-terra (short context) Data Zone	N/A	Input: $5.50
Cached Input: $0.55
Cache writes: $6.88
Output: $33

Model |Input | Cached Input | Cache Writes | Output |
|---|---|---|---|--|
|GPT-5.6-terra (long context) Global| $5 | $0.50 | $6.25| $22.50|