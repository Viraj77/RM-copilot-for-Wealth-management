# Knowledge Base — Source Documents for Relationship Manager Copilot (Capstone P6)

This package contains 10 source documents spanning the three required
categories (product guides, suitability/compliance policy, research notes)
and four file formats (PDF, DOCX, TXT, CSV), as required by Section 4
("Ingestion Pipeline") of the capstone brief. **Every document is 10+ pages
(or row-equivalent for the CSVs) of unique content — no repetition across
documents.**

## Document Inventory

| # | Doc ID  | Category        | Title                                             | Format | Length | File |
|---|---------|-----------------|----------------------------------------------------|--------|--------|------|
| 1 | PG-001  | Product Guide   | Horizon Balanced Growth Fund (HBGF)                | DOCX   | 10 pages | product_guides/PG-001_Balanced_Growth_Fund.docx |
| 2 | PG-002  | Product Guide   | Horizon Conservative Income Fund (HCIF)            | TXT    | 11 pages | product_guides/PG-002_Conservative_Income_Fund.txt |
| 3 | PG-003  | Product Guide   | Horizon Aggressive Equity Fund (HAEF)              | PDF    | 10 pages | product_guides/PG-003_Aggressive_Equity_Fund.pdf |
| 4 | PG-004  | Product Guide   | Fixed Income & Structured Products (FD/Bond Ladder)| CSV    | 160 rows × 17 cols | product_guides/PG-004_Fixed_Income_Structured_Products.csv |
| 5 | CMP-001 | Compliance      | Suitability & Compliance Policy                    | DOCX   | 10 pages | compliance/CMP-001_Suitability_Compliance_Policy.docx |
| 6 | CMP-002 | Compliance      | Restricted List & Entitlement Rules                | CSV    | 83 rows × 13 cols | compliance/CMP-002_Restricted_List_Entitlement_Rules.csv |
| 7 | CMP-003 | Compliance      | Client Risk Profiling Methodology                  | PDF    | 10 pages | compliance/CMP-003_Client_Risk_Profiling_Methodology.pdf |
| 8 | RN-001  | Research Note   | Q2 2026 Global Market Outlook                      | PDF    | 10 pages | research/RN-001_Q2_2026_Market_Outlook.pdf |
| 9 | RN-002  | Research Note   | Sector Deep Dive — Technology & Applied AI         | TXT    | 10 pages | research/RN-002_Sector_Deep_Dive_Tech_AI.txt |
| 10| RN-003  | Research Note   | Fixed Income Outlook — Mid-Year 2026               | DOCX   | 10 pages | research/RN-003_Fixed_Income_Outlook.docx |

**Format distribution:** 3 DOCX · 2 TXT · 3 PDF · 2 CSV
**Category distribution:** 4 Product Guides · 3 Compliance/Policy · 3 Research Notes
**Total length:** 80+ pages across the 8 PDF/DOCX/TXT documents, plus 243
combined structured data rows across the two CSVs — each document
independently clears the 10-page (or equivalent) bar with zero content
repeated across documents.

## How the documents interlock (useful for your RAG/agent demo)

These aren't independent — they're written with deep cross-references to
support multi-hop retrieval testing (Part B, "Compare single-shot vs
multi-hop retrieval"):

- **PG-001/002/003** each define detailed `suitable_risk_profiles`
  constraints, sub-strategy breakdowns, stress-test scenarios, and
  scenario-based suitability walkthroughs that **CMP-001** (Suitability
  Policy) and **CMP-003** (Risk Profiling Methodology) operationalize into
  a scoring/gating process.
- **CMP-002** (Restricted List, 83 rules) contains granular per-product-family
  rules (e.g. `RULE-002` sector concentration, `RULE-003` structured note
  profile gate, plus dozens of per-fund and per-tenor operational controls)
  that directly reference products in **PG-001/003/004**.
- **RN-001** (Q2 Market Outlook) cross-references **RN-002** (Tech sector
  deep dive) and **RN-003** (Fixed Income outlook) repeatedly — including a
  dedicated "Comparison to Prior Quarter Outlook" section — and ties
  positioning views back to **PG-001/002/003** by name.
- **RN-002**'s 17-section deep dive (supply chain, energy infrastructure,
  regulatory landscape, bull/bear scenarios) is built to be retrieved
  alongside **PG-003**'s sector-exposure tables for combined product +
  research queries.
- **PG-004** and **CMP-002** are both large CSVs (160 and 83 rows
  respectively), so you can test structured metadata-filtered retrieval
  (e.g., filter by `risk_rating`, `rule_category`, `product_category`) vs.
  unstructured chunk retrieval on the prose documents.

## Metadata tagging (for ingestion)

Each document's header/front-matter (or CSV columns) carries the metadata
fields your ingestion pipeline expects per Section 3 of the brief:
`doc_id`, `type`, `date`, `source`, `sensitivity`. RN-002 and the rules in
CMP-002 are tagged at a sensitivity tier above "Public" — useful for testing
your entitlement-filtering scenario ("Restricted research for an
unentitled RM → filtered out").

## Suggested test scenarios this set supports

- "Prepare talking points for a Balanced client reviewing HBGF" → PG-001 + CMP-001
- "Is the Structured Capital Note suitable for a Conservative client?" → PG-004 + CMP-002 (RULE-003) → Blocked
- "Summarize portfolio risk for a client holding HAEF" → PG-003 risk/stress-test sections
- "What's the house view on fixed income duration right now?" → RN-001 + RN-003 (multi-hop)
- "Walk me through how a client's risk tier is calculated" → CMP-003 scoring methodology + worked examples
- Restricted research request from an unentitled RM → RN-002 (Tier 2 gate) filtered per CMP-001 Section 7
- "Compare HBGF, HCIF, and HAEF" → tests retrieval across all three fund product guides plus the cross-comparison sections each guide contains

## Note on synthetic data

All entities (fund names, AUM figures, performance numbers, policy rules)
are fictional and created for this capstone exercise. No real fund,
institution, or client data is represented.
