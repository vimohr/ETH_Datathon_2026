# Geo-Insight: Which Crises Are Most Overlooked?

## Overview
In this challenge, you will build a system that surfaces mismatches between humanitarian need and humanitarian financing coverage across active crises worldwide.

Your task is to take a crisis context, a geographic scope, or a natural-language query and return the situations that are most underserved, ranked by the gap between actual need and available funding.

This challenge is based on a real analytical problem inside the humanitarian data ecosystem. Humanitarian coordinators and donor advisors need to quickly identify where funds are not reaching, relative to the scale of a crisis.

---

## The Problem
Humanitarian data mixes two very different kinds of signals:
* **Objective severity indicators** that describe the scale and urgency of a crisis.
* **Funding and coverage data** that describes what has actually been resourced.

**Example questions a decision-maker might ask:**
* *"Which crises have the highest proportion of people in need but the lowest fund allocations?"*
* *"Are there countries with active HRPs where funding is absent or negligible?"*
* *"Which regions are consistently underfunded relative to need across multiple years?"*
* *"Show me acute food insecurity hotspots that have received less than 10% of their requested funding."*

In all four examples, some signals are objective thresholds and some are relative or contextual judgments. Your system should separate these layers and combine them effectively.

---

## Core Task
Given a query or geographic scope:

1.  **Identify** relevant crises or countries using severity and needs data.
2.  **Filter** to situations meeting a meaningful threshold of documented need.
3.  **Interpret** funding coverage data to compute a gap or mismatch score.
4.  **Rank** the crises by how overlooked they appear, relative to need.

**Your result should ideally support:**
* A ranked list of crises or countries with a gap score or coverage ratio.
* Map-ready outputs using country or crisis coordinates where available.
* A short explanation of why the top results rank as most overlooked.

> **Note:** Assume you only have publicly available datasets and the current query or scope provided.

---

## Bonus Task
Use temporal or cross-source signals to improve ranking and identify structural neglect rather than just point-in-time gaps.

**Examples of additional signals:**
* Multi-year funding trends for the same crisis.
* HRP target vs. actual coverage over time.
* Whether a crisis appears in global media or advocacy reporting.
* Population displacement or IDP figures as a need multiplier.
* Sector-level gaps within a crisis (e.g., health vs. food vs. shelter).
* Donor concentration, where a crisis relies on one or two major donors.

**Bonus question:** How should ranking change when a crisis has been underfunded for multiple consecutive years versus one that is newly underfunded? How can you represent structural issues differently from acute emergencies?

---

## Directions You Can Explore
You are free to choose different solution styles as long as the core task is addressed and the final outcome is strong.

**Possible directions include:**
* A gap-scoring pipeline using HNO and funding data only.
* A retrieval system over crisis summaries and metadata.
* An LLM-assisted query understanding layer that maps natural-language questions to filter criteria.
* A hybrid approach combining structured funding ratios with semantic scoring over crisis descriptions.
* Geospatial analysis using country centroids or crisis coordinates.
* Enrichment using external data such as ACLED conflict events, IPC food security phases, or UNHCR displacement figures.
* Time-series analysis of funding trends per crisis.
* A lightweight visualization or dashboard.
* A conversational interface where a user can refine scope across multiple turns.

Everything that helps answer the core question of where need outpaces coverage is encouraged.

> **Note:** The core judging focus is the quality and defensibility of the gap ranking and the breadth of crisis types and queries your system can handle well.

---

## Data Provided
You are provided with links to the following publicly available datasets as a starting point:

* **Humanitarian Needs Overview data** (includes people in need figures by country and sector): [https://data.humdata.org/dataset/global-hpc-hno](https://data.humdata.org/dataset/global-hpc-hno)
* **Humanitarian Response Plan data** (includes funding targets and plan status): [https://data.humdata.org/dataset/humanitarian-response-plans](https://data.humdata.org/dataset/humanitarian-response-plans)
* **Global common operational datasets for population:** [https://data.humdata.org/dataset/cod-ps-global](https://data.humdata.org/dataset/cod-ps-global)
* **Global requirements and funding data** (includes overall financial tracking): [https://data.humdata.org/dataset/global-requirements-and-funding-data](https://data.humdata.org/dataset/global-requirements-and-funding-data)
* **CBPF Pooled Funds Data Hub** (includes country-based pooled fund allocations and visualizations): [https://cbpf.data.unocha.org/](https://cbpf.data.unocha.org/)

You are encouraged to supplement these with additional public sources where it improves your analysis. Declare any external data sources you use.

---

## What You Need to Build
You are free to choose the format of your solution, as long as it clearly solves the problem.

**Examples:**
* A gap analysis and ranking pipeline
* An API or service that accepts a query and returns ranked crises
* A notebook-based prototype
* An interactive dashboard or map
* A conversational interface

The primary focus is analytical quality and ranking defensibility, not frontend polish.

---

## Expected Output
Your system should return crises or countries that:
* Meet a meaningful threshold of documented humanitarian need.
* Are ordered by the size or severity of the mismatch between need and funding coverage.

**Outputs should at least include:**
* A ranked list of crises or countries with a gap score or coverage ratio.

**It may also include:**
* The filters or thresholds applied to define in-scope crises.
* The scoring logic at a high level.
* Brief explanations for why the top-ranked crises appear most overlooked.

---

## Solution Ideas
There is no single correct approach. Strong solutions may include one or more of the following:

* A computed gap ratio using people in need versus funding received or allocated.
* Threshold-based filtering on Humanitarian Response Plan status or minimum need size.
* LLM-based query decomposition to extract scope, severity floor, and funding ceiling from natural-language questions.
* Embedding-based retrieval over crisis summary text.
* Geospatial scoring to cluster or compare crises by region.
* Enrichment from conflict, displacement, or food security data to validate or amplify need signals.
* Per-crisis explanation generation suitable for a briefing note or decision memo.

You may use external models or APIs, but you must clearly document them.

---

## Design Challenges
This problem is interesting because humanitarian data is messy, inconsistently structured, and politically sensitive.

**Examples of things your system may need to handle:**
* Crises with partial or outdated Humanitarian Needs Overview figures.
* Countries with active need but no formal Humanitarian Response Plan in place.
* Humanitarian funding allocations that lag behind needs assessments.
* The difference between funding requested, funding pledged, and funding disbursed.
* Situations where a large total funding figure masks severe sector-level gaps.
* Crises that are well-funded in aggregate but have specific population groups that are consistently missed.

---

## Deliverables
**Submit the following:**
* A working prototype.
* A short technical write-up.
* A demo, notebook, API, or dashboard that shows the full result flow.
* A description of your gap-scoring and ranking logic.
* A short discussion of failure cases, limitations, or open problems.

**Your technical write-up should cover:**
* How you define and compute the mismatch or gap score.
* How you handle missing, outdated, or inconsistent data.
* Which models, heuristics, or external systems you use.
* How you would extend the system with temporal or cross-source signals for the bonus task.

---

## Evaluation
Submissions will be evaluated using a mix of automatic scoring and human judgment.

**Automatic evaluation may assess:**
* Whether hard need thresholds are correctly applied as filters.
* Coverage ratio accuracy against ground-truth funding figures.
* Ranking consistency across equivalent queries.
* How gracefully the system handles crises with incomplete data.

**Jury review will look at:**
* Correctness and robustness of the gap analysis.
* Relevance and defensibility of the ranking.
* Originality of the approach.
* Practical usefulness for a humanitarian analyst or donor advisor.
* Quality of the demo and presentation.

Peer feedback may also play a role, particularly around perceived trust in the results and clarity of the explanations.

---

## Rules
* Use the provided datasets as the primary source of truth for need and funding figures.
* You may use external models and APIs, but you must declare them.
* Your submission should be understandable and reproducible at a high level.
* Your system should handle ambiguous or underspecified queries gracefully.
* **Do not fabricate or hallucinate funding or need figures.** Ground all outputs in the provided data.

---

## Tips
* Hybrid approaches combining structured ratios with contextual signals tend to outperform single-method systems.
* Make your ranking logic explainable enough that a humanitarian analyst could defend it in a briefing.
* If you attempt the bonus task, think carefully about how to distinguish a crisis that has always been overlooked from one that was recently well-funded but has since deteriorated.

---

## Success Criteria
A strong submission should:
* Generate outputs that a non-technical humanitarian coordinator could act on.
* Be honest about what it does not know, surface uncertainty clearly, and avoid presenting gap scores with false precision.
* Be designed for decision support, not automated decision-making, and demonstrate awareness of the difference.

The strongest submissions will not just produce a ranking. They will also clearly articulate how the tool would fit into a real workflow, who would use it, what they would do next after seeing the results, and what could go wrong if the outputs were misread or misapplied.

---

## Final Brief
Given a query or geographic scope, find the crises with the most significant mismatch between documented humanitarian need and pooled fund coverage, and rank them by how overlooked they appear. Bonus: extend the ranking to capture structural or chronic neglect using multi-year or cross-source signals.

> **As you dive in, keep one important point in mind:** humanitarian data represents people, often people in extremely vulnerable situations. Good humanitarian data work does not just optimize numbers. It respects context.
>
> These challenges do not have a single right answer. What we are looking for are thoughtful approaches, transparent assumptions, and clear reasoning. The goal is tools that could help decision-makers ask better questions and make better-informed choices, not tools that make the choices for them.