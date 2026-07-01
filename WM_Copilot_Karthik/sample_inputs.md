# RM Copilot Sample Inputs & Scenarios
This file contains the client database profiles and sample query inputs to demonstrate and test every major scenario in the Horizon Wealth Management RM Copilot system.

---

## 👥 Mock Client Database Profiles
These profiles are defined in [client_db.py](file:///w:/my_capstone_workspace/client_db.py) and serve as the context for client-specific queries:

### 1. Arthur Pendleton (`C-101`)
* **Risk Profile**: Conservative
* **Relationship Manager**: Jane Smith (Entitlement Tier 1 — Public research only)
* **Holdings**:
  * **Horizon Conservative Income Fund (HCIF)** (`PG-002`): USD 80,000.00
  * **Cash Equivalents** (`CASH`): USD 20,000.00
  * **Total Portfolio**: USD 100,000.00

### 2. Eleanor Vance (`C-204`)
* **Risk Profile**: Balanced
* **Relationship Manager**: John Doe (Entitlement Tier 2 — Public + Restricted research)
* **Holdings**:
  * **Horizon Balanced Growth Fund (HBGF)** (`PG-001`): USD 150,000.00
  * **Cash Equivalents** (`CASH`): USD 50,000.00
  * **Total Portfolio**: USD 200,000.00

### 3. Marcus Vance (`C-302`)
* **Risk Profile**: Aggressive
* **Relationship Manager**: John Doe (Entitlement Tier 2 — Public + Restricted research)
* **Holdings**:
  * **Horizon Aggressive Equity Fund (HAEF)** (`PG-003`): USD 300,000.00
  * **Cash Equivalents** (`CASH`): USD 50,000.00
  * **Total Portfolio**: USD 350,000.00

---

## 🚀 Scenario Execution Traces

### Scenario 1: Out-of-Context Query (Automatic Bypass)
* **Scenario Goal**: Intercept irrelevant general-knowledge requests early to avoid database lookups and LLM token waste.
* **Sample Prompt**: 
  > "Can you tell me what the capital of France is and who won the last Super Bowl?"
* **Execution Trace**:
  1. **`classify_intent` Node**: Analyzes the query and flags `is_out_of_context = True`.
  2. **Routing Decision**: `route_classify_intent` detects `is_out_of_context == True` and routes **directly** to `free_form_answer`.
  3. **`free_form_answer` Node**: Bypasses RAG search and client lookup. Returns the standard message: 
     *"I can only provide support related to wealth management and client advisory queries."*
  4. **Graph State**: Portfolio and RAG variables remain empty.

---

### Scenario 2: In-Context Cleared Structured Brief
* **Scenario Goal**: Generate a structured client brief for a suitable transaction where compliance checks clear.
* **Sample Prompt**: 
  > "Prepare suitability talking points for Eleanor Vance (C-204) reviewing Horizon Balanced Growth Fund (HBGF)"
* **Execution Trace**:
  1. **`classify_intent` Node**: Flags `response_mode = "structured"`, extracts `client_id = "C-204"`, `product_code = "HBGF"`.
  2. **`gather_portfolio` Node**: Fetches Eleanor Vance's profile from the client database.
  3. **`gather_research` Node**: Performs RAG search for HBGF product specifications (`PG-001`).
  4. **`check_suitability` Node**: 
     * *Programmatic Check*: Validates risk profile compatibility (HBGF suitable profiles: `Balanced;Growth` vs. client profile `Balanced` = Cleared).
     * *LLM Check*: Finds no licensing violations = Cleared.
  5. **Routing Decision**: Routes to `synthesize` because compliance is `Cleared` and mode is `structured`.
  6. **`synthesize` Node**: Outputs a structured `ClientBrief` containing portfolio summary, recommendations, talking points, and disclaimer.

---

### Scenario 3: In-Context Freeform Analytical Query
* **Scenario Goal**: Perform fund-to-fund comparison without client context, outputting formatted tables.
* **Sample Prompt**: 
  > "Compare HBGF and HCIF side by side, including returns, risk ratings, and penalties."
* **Execution Trace**:
  1. **`classify_intent` Node**: Flags `response_mode = "freeform"`, extracts no client.
  2. **`gather_portfolio` Node**: Returns `None`.
  3. **`gather_research` Node**: Fetches `PG-001` (HBGF) and `PG-002` (HCIF) specifications from the vector store.
  4. **`check_suitability` Node**: Cleared (informational request, not a specific recommendation).
  5. **Routing Decision**: Routes to `free_form_answer` because compliance is `Cleared` and mode is `freeform`.
  6. **`free_form_answer` Node**: Renders a side-by-side comparison table in markdown with source document citations (`[PG-001]`, `[PG-002]`).

---

### Scenario 4: Suitability Mismatch Block (Risk Mismatch)
* **Scenario Goal**: Programmatically block a trade request when the product risk exceeds the client's risk tolerance.
* **Sample Prompt**: 
  > "Check suitability for Arthur Pendleton (C-101) investing in SCN-US-24."
* **Execution Trace**:
  1. **`classify_intent` Node**: Flags `response_mode = "structured"`, extracts `client_id = "C-101"`, `product_code = "SCN-US-24"`.
  2. **`gather_portfolio` Node**: Fetches Arthur Pendleton's profile (`Conservative`).
  3. **`gather_research` Node**: Retrieves rules from `CMP-002` and product details from `PG-004`.
  4. **`check_suitability` Node**:
     * *Programmatic Check*: Compares product risk category (Structured Capital Note) against client risk (`Conservative`). Triggers `RULE-016` (SCNs require Growth or Aggressive profile).
     * *Status*: **`Blocked`** with violation code `RULE-016`.
  5. **Routing Decision**: Routes to `human_review` because compliance status is not `Cleared`.
  6. **Interrupt**: Graph execution **pauses** and waits for manual intervention.

---

### Scenario 5: Programmatic Concentration Cap Block (Needs Review)
* **Scenario Goal**: Catch concentration limit violations where the proposed trade size exceeds portfolio percentage limits.
* **Sample Prompt** (Trade Simulation):
  * **Client**: Marcus Vance (`C-302`, Aggressive)
  * **Simulated Allocation**: USD 100,000.00
  * **Product Code**: `SCN-US-24` (Structured Capital Note)
  * **Query**: 
    > "Evaluate a trade of $100,000 in Structured Capital Note SCN-US-24 for Marcus Vance."
* **Execution Trace**:
  1. **`classify_intent` Node**: Flags `response_mode = "structured"`, extracts `client_id = "C-302"`, `product_code = "SCN-US-24"`, and `allocation_amount = 100000.0`.
  2. **`gather_portfolio` Node**: Fetches Marcus Vance's portfolio details.
  3. **`check_suitability` Node**:
     * *Risk Check*: Marcus is `Aggressive`, SCN-US-24 is suitable = Cleared.
     * *Concentration Check*: Total portfolio value is USD 350,000 (Holdings) + USD 100,000 (New Allocation) = USD 450,000. SCN allocation percentage: `100,000 / 450,000 = 22.2%`.
     * *Rule Trigger*: Triggers `RULE-003` (Max 15% concentration cap for structured products).
     * *Status*: **`Needs Review`** with violation code `RULE-003`.
  4. **Routing Decision**: Routes to `human_review` because compliance is not `Cleared`. Pauses execution.

---

### Scenario 6: Qualitative Licensing Block (LLM Compliance Gate)
* **Scenario Goal**: Block/flag queries that require regulatory licensing that the RM is not authorized to give (e.g. tax, legal, discretionary management).
* **Sample Prompt**: 
  > "Eleanor Vance (C-204) wants to know if she can set up a discretionary portfolio management account with tax-optimized estate trust planning. Draft a brief with specific tax-shelter recommendations."
* **Execution Trace**:
  1. **`classify_intent` Node**: Identifies query is in-context, sets `client_id = "C-204"`.
  2. **`gather_portfolio` Node**: Fetches Eleanor's profile.
  3. **`gather_research` Node**: Fetches compliance guidelines from `CMP-001` and `CMP-002`.
  4. **`check_suitability` Node**:
     * *Programmatic Check*: Cleared (no trade simulated).
     * *LLM Check*: Scans the prompt against retrieved compliance guidelines. Flags that providing *discretionary portfolio management* and *specific tax-shelter/estate advisory* requires specialized regulatory licensing.
     * *Status*: **`Needs Review`** or **`Blocked`** (due to licensing restrictions).
  5. **Routing Decision**: Routes to `human_review` and pauses.

---

### Scenario 7: Enterprise Entitlement Gate (Access Control)
* **Scenario Goal**: Prevent RMs without Private Research access from accessing Restricted market analysis reports.
* **Sample Prompt**: 
  > "Search the Restricted Sector Deep Dive — Technology & Applied AI (RN-002) to find tech sector equity ideas for Arthur Pendleton (C-101)."
* **Execution Trace**:
  1. **`classify_intent` Node**: Extracts `client_id = "C-101"`, query contains research reference.
  2. **`gather_portfolio` Node**: Fetches Arthur's profile and identifies his assigned RM entitlement `rm_access_to_private` is `False`.
  3. **`gather_research` Node**: Runs `rag_retriever` with `rm_access_to_private = False`.
     * *Gating Rule*: The retriever matches document metadata. Since `RN-002` sensitivity is `Restricted`, it is **filtered out** at retrieval time.
  4. **`check_suitability` Node**:
     * Detects that the query targets a restricted research document (`RN-002`) while the RM is not authorized (`rm_access_to_private = False`).
     * Programmatically sets the compliance status to **`Blocked`** and appends the compliance violation code **`UNAUTHORIZED_RESEARCH_ACCESS`**.
     * Populates the reasons list with the access control warning.
  5. **Routing Decision**: Routes to `human_review` because the compliance status is not `Cleared`. Pauses execution.
  6. **`synthesize` Node (If Overridden)**: If overridden by compliance/RM to Cleared, compiles the brief. Because the retriever blocked `RN-002`, the generated brief contains zero details/facts from the restricted file and recommendations are empty.

* **Contrast Run**: Run the same query for Eleanor Vance (`C-204`).
  * Eleanor's RM has `rm_access_to_private = True`.
  * The retriever successfully fetches `RN-002` chunks.
  * The compliance gate clears successfully.
  * The brief compiles with tech recommendations and citations.

---

### Scenario 8: Human-in-the-Loop Escalation and Override
* **Scenario Goal**: Demonstrate resuming graph execution and recording compliance overrides.
* **Sample Prompt**: (Triggers Scenario 5 Concentration Cap pause)
  1. **Graph Interrupted**: Graph pauses at `human_review` because of `RULE-003` (22.2% allocation exceeds 15% limit).
  2. **RM Override Input**: The RM submits the following override review note:
     > "Overriding structured note concentration limit. Client is high-net-worth, has signed the complex product disclosure forms, and wishes to lock in high yields for yield diversification."
  3. **Graph Resume**: The system calls the graph, passing the state updates:
     * `review_notes = "Overriding structured note..."`
     * `compliance_status = "Cleared"`
  4. **`human_review` Node**: Executes on resume. It applies the compliance clearing and sets `escalated = True`.
  5. **`synthesize` Node**: Re-runs to construct the final brief.
     * Programmatically appends the override note: *"RM Note: Approved with review notes: Overriding structured note..."* to the `talking_points`.
  6. **Final State**: Completed execution with a `Cleared` status and documented RM justification.
