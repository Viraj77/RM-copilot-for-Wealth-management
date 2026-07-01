import os
import json
import pandas as pd
from client_db import get_client_profile
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "wealth_mgmt_knowledge"
PG004_CSV_PATH = "docs/product_guides/PG-004_Fixed_Income_Structured_Products.csv"
CMP002_CSV_PATH = "docs/compliance/CMP-002_Restricted_List_Entitlement_Rules.csv"

def portfolio_lookup(client_id: str):
    """
    Look up client name, risk profile, current holdings, and assigned relationship manager.
    """
    profile = get_client_profile(client_id)
    if not profile:
        return {"error": f"Client ID '{client_id}' not found."}
    return profile

def market_data_tool(product_code: str):
    """
    Look up market details and product parameters for a given product code.
    """
    if not product_code:
        return {"error": "No product code provided."}
    product_code = product_code.upper()
    
    # Core Funds check
    if product_code == "PG-001" or product_code == "HBGF":
        return {
            "product_code": "PG-001",
            "product_name": "Horizon Balanced Growth Fund (HBGF)",
            "product_category": "Mutual Fund - Hybrid",
            "indicative_return": "7.2% p.a. (historical 3-yr average)",
            "risk_rating": "3 (Moderate)",
            "suitable_risk_profiles": "Balanced;Growth",
            "early_withdrawal_penalty": "None (Redemption T+3)",
            "capital_protection": "None",
            "liquidity": "Daily liquidity",
            "details": "Slightly growth-oriented, holds 50-70% equities, 30-50% bonds/cash. Subject to moderate market risk."
        }
    elif product_code == "PG-002" or product_code == "HCIF":
        return {
            "product_code": "PG-002",
            "product_name": "Horizon Conservative Income Fund (HCIF)",
            "product_category": "Mutual Fund - Hybrid (Conservative)",
            "indicative_return": "3.9% annual yield",
            "risk_rating": "2 (Low-Moderate)",
            "suitable_risk_profiles": "Conservative;Balanced",
            "early_withdrawal_penalty": "1.0% if redeemed within 180 days; Nil thereafter",
            "capital_protection": "None",
            "liquidity": "Daily liquidity",
            "details": "Preservation-focused, holds 75-90% high-grade fixed income, 10-25% defensive low-volatility equities."
        }
    elif product_code == "PG-003" or product_code == "HAEF":
        return {
            "product_code": "PG-003",
            "product_name": "Horizon Aggressive Equity Fund (HAEF)",
            "product_category": "Mutual Fund - Equity",
            "indicative_return": "12.4% p.a. (historical 3-yr average)",
            "risk_rating": "5 (High)",
            "suitable_risk_profiles": "Aggressive",
            "early_withdrawal_penalty": "None (Redemption T+3)",
            "capital_protection": "None",
            "liquidity": "Daily liquidity",
            "details": "Growth-focused, holds 90%+ global equities, significant sector tilts (e.g. 28-38% tech/applied AI)."
        }
        
    # Fixed Income & Structured Products check
    if os.path.exists(PG004_CSV_PATH):
        try:
            df = pd.read_csv(PG004_CSV_PATH)
            match = df[df['product_code'].str.upper() == product_code]
            if not match.empty:
                rec = match.iloc[0].to_dict()
                return {
                    "product_code": rec.get("product_code"),
                    "product_name": rec.get("product_name"),
                    "product_category": rec.get("product_category"),
                    "indicative_return": rec.get("indicative_rate_pct"),
                    "risk_rating": str(rec.get("risk_rating")),
                    "suitable_risk_profiles": rec.get("suitable_risk_profiles"),
                    "early_withdrawal_penalty": rec.get("early_withdrawal_penalty"),
                    "capital_protection": rec.get("capital_protection"),
                    "liquidity": rec.get("liquidity"),
                    "details": f"Tenor: {rec.get('tenor_months')} months, Min Investment: USD {rec.get('min_investment_usd')}, Issuer Rating: {rec.get('issuer_credit_rating')}"
                }
        except Exception as e:
            return {"error": f"Error searching product CSV: {e}"}
            
    return {"error": f"Product Code '{product_code}' not found in approved shelf."}

def suitability_checker(client_id: str, product_code: str, allocation_amount: float = 0.0):
    """
    Programmatic and LLM-assisted compliance check comparing a product and trade details to a client profile.
    Checks:
    1. Product suitability list vs Client Risk Profile.
    2. Restricted list rules (like Structured Product Caps / concentration rules).
    """
    import re
    from typing import List, Literal
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI

    client = get_client_profile(client_id)
    if not client:
        return {"status": "Blocked", "reason": f"Client ID '{client_id}' not found.", "violations": ["Client not found"]}

    # Guard: no product code means no programmatic product check can be run
    if not product_code:
        return {"status": "Cleared", "violations": [], "reasons": ["No product code specified — programmatic suitability check skipped."], "citations": []}

    client_risk = client["risk_profile"]
    product = market_data_tool(product_code)
    
    if "error" in product:
        return {"status": "Blocked", "reason": product["error"], "violations": ["Product not found"]}

    # Programmatic fallback implementation in case LLM fails
    def programmatic_fallback():
        violations = []
        reasons = []
        citations = []
        
        suitable_profiles = [x.strip() for x in product["suitable_risk_profiles"].split(";")]
        if client_risk not in suitable_profiles:
            violations.append("INCOMPATIBLE_RISK_PROFILE")
            reasons.append(f"Product '{product_code}' is suitable for {product['suitable_risk_profiles']} but client risk profile is '{client_risk}'.")
            citations.append(f"{product['product_code']} Product Guide")
            
        holdings = client["holdings"]
        total_val = sum(h["allocation_amount"] for h in holdings) + allocation_amount
        
        if os.path.exists(CMP002_CSV_PATH):
            try:
                df = pd.read_csv(CMP002_CSV_PATH)
                if "SCN" in product_code or product["product_category"] == "Structured Product":
                    if client_risk in ["Conservative", "Balanced"]:
                        match_rule = df[df['rule_id'] == 'RULE-016']
                        if not match_rule.empty:
                            rule = match_rule.iloc[0]
                            violations.append(rule['rule_id'])
                            reasons.append(f"{rule['rule_id']}: {rule['condition']}")
                            citations.append("CMP-002 (RULE-016)")
                            
                    if total_val > 0:
                        pct = (allocation_amount / total_val) * 100
                        if pct > 15:
                            match_rule = df[df['rule_id'] == 'RULE-003']
                            if not match_rule.empty:
                                rule = match_rule.iloc[0]
                                violations.append(rule['rule_id'])
                                reasons.append(f"{rule['rule_id']}: Allocation of {pct:.1f}% exceeds 15% structured product cap.")
                                citations.append("CMP-002 (RULE-003)")
            except Exception as e:
                print(f"Error checking suitability CSV: {e}")
                
        if any(v in ["RULE-016", "INCOMPATIBLE_RISK_PROFILE"] for v in violations):
            status = "Blocked"
        elif violations:
            status = "Needs Review"
        else:
            status = "Cleared"
            
        return {
            "status": status,
            "violations": violations,
            "reasons": reasons,
            "citations": list(set(citations))
        }

    # Retrieve all compliance documents from Chroma DB using semantic search
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # Build semantic search query based on doc type, risk profile, and category details
        search_query = f"compliance restriction rules, concentration limits, and suitability caps for {client_risk} risk profile client buying {product_code} ({product.get('product_category')})"
        results = db.similarity_search(search_query, k=40, filter={"type": "compliance"})
        
        unique_rules = {}
        other_policy_chunks = []
        seen_contents = set()
        
        for doc in results:
            text = doc.page_content
            match_rule = re.search(r'rule_id:\s*(RULE-\d+)', text)
            if match_rule:
                rule_id = match_rule.group(1)
                unique_rules[rule_id] = text
            else:
                if text not in seen_contents:
                    seen_contents.add(text)
                    other_policy_chunks.append(text)
                    
        rules_context = "\n\n".join(unique_rules.values())
        policy_context = "\n\n".join(other_policy_chunks)
        
        # Pre-calculate portfolio statistics to prevent LLM math mistakes
        current_portfolio_value = sum(h["allocation_amount"] for h in client["holdings"])
        new_portfolio_value = current_portfolio_value + allocation_amount
        proposed_concentration_pct = (allocation_amount / new_portfolio_value * 100) if new_portfolio_value > 0 else 0.0
        
        holdings_lines = []
        for h in client["holdings"]:
            holdings_lines.append(f"- {h['product_name']} ({h['product_code']}): USD {h['allocation_amount']:,} [{h['asset_class']}]")
        holdings_text = "\n".join(holdings_lines)
        
        prompt = f"""
        You are the Compliance and Suitability Engine for Horizon Wealth Management.
        Your task is to evaluate a proposed transaction against the client's profile and the retrieved compliance rules and guidelines.

        === PROPOSED TRANSACTION ===
        - Client ID: {client_id}
        - Client Name: {client.get('name')}
        - Client Risk Profile: {client_risk}
        - Product Code to Buy: {product_code}
        - Product Name: {product.get('product_name')}
        - Product Category: {product.get('product_category')}
        - Product Risk Rating: {product.get('risk_rating')}
        - Suitable Risk Profiles for Product: {product.get('suitable_risk_profiles')}
        - Proposed Allocation Amount: USD {allocation_amount:,.2f}

        === CLIENT PORTFOLIO STATS (PRE-CALCULATED) ===
        - Total Current Holdings Value: USD {current_portfolio_value:,.2f}
        - New Total Portfolio Value (including proposed allocation): USD {new_portfolio_value:,.2f}
        - Proposed Product Concentration in Portfolio: {proposed_concentration_pct:.2f}%
        
        === CLIENT CURRENT HOLDINGS ===
        {holdings_text}

        === RETRIEVED COMPLIANCE RULES & GUIDELINES ===
        {rules_context}
        {policy_context}

        === EVALUATION INSTRUCTIONS ===
        For each retrieved rule or policy:
        1. **Check Applicability**:
           - Does the rule apply to this product code, name, or category? (For example, rules about Structured Products or SCNs only apply if the product category is "Structured Product" or the code contains "SCN").
           - Does the rule apply to this client or transaction type?

        2. **Evaluate Violations**:
           - **Risk Profile Compatibility**:
             - Check if the client's risk profile ('{client_risk}') is compatible with the product. The product's suitable risk profiles are '{product.get('suitable_risk_profiles')}'. If the client's risk profile is not in the suitable list, add 'INCOMPATIBLE_RISK_PROFILE' to violations.
             - Also check if any retrieved profile restriction rules apply (e.g. if a retrieved rule says Structured Capital Notes (SCN series) require a Growth or Aggressive profile, check if the client risk profile is Conservative or Balanced. If it is, the trade is Blocked. You must add the corresponding rule ID (e.g. 'RULE-016') and 'INCOMPATIBLE_RISK_PROFILE' to the violations list).
           - **Concentration Limits**:
             - If a concentration limit rule applies (e.g. a rule with concentration cap like RULE-003 structured note concentration cap), check if the pre-calculated concentration ({proposed_concentration_pct:.2f}%) exceeds the percentage limit specified in the rule (for RULE-003, the limit is 15.0%). If it does, add the corresponding rule ID (e.g. 'RULE-003') to violations.

        3. **Determine Overall Status**:
           - "Blocked": If any rule is violated that represents a hard block (e.g. INCOMPATIBLE_RISK_PROFILE or profile restriction rule like RULE-016).
           - "Needs Review": If any concentration limit or review rules are violated (e.g. RULE-003 concentration breach), but no hard blocks are violated.
           - "Cleared": If no compliance rules or suitability limits are violated.

        4. **Format Output**:
           - `status`: One of "Cleared", "Needs Review", "Blocked".
           - `violations`: List of rule IDs violated (e.g., ['RULE-016', 'INCOMPATIBLE_RISK_PROFILE'] or ['RULE-003']). Return empty list [] if no rules are violated.
           - `reasons`: Detailed explanation for each violation or why it passed, showing calculations for concentration percentages.
           - `citations`: List cited rules as 'CMP-002 (RULE-XXX)' (e.g. 'CMP-002 (RULE-016)') or '[product_code] Product Guide' (e.g. '{product_code} Product Guide').

        Perform the evaluation and output the structured result.
        """
        
        class SuitabilityReport(BaseModel):
            status: Literal["Cleared", "Needs Review", "Blocked"]
            violations: List[str] = Field(description="List of rule IDs violated (e.g. ['RULE-016', 'INCOMPATIBLE_RISK_PROFILE'] or ['RULE-003']). Return empty list [] if no rules are violated.")
            reasons: List[str] = Field(description="Detailed reasons explaining why each rule was violated or passed, including percentage calculations for concentration rules.")
            citations: List[str] = Field(description="List of cited documents or rules, e.g. ['CMP-002 (RULE-016)', 'PG-001 Product Guide'].")
            
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        structured_llm = llm.with_structured_output(SuitabilityReport)
        report = structured_llm.invoke(prompt)
        
        return {
            "status": report.status,
            "violations": report.violations,
            "reasons": report.reasons,
            "citations": list(set(report.citations))
        }
    except Exception as e:
        print(f"LLM Suitability Checker error, falling back to programmatic check: {e}")
        return programmatic_fallback()

def rag_retriever(query: str, rm_access_to_private: bool = False):
    """
    Vector database retriever with metadata filtering for RM entitlement,
    fully dynamic without hardcoded document IDs or keyword maps in python.
    rm_access_to_private:
      - False: Public documents only
      - True: Public + Restricted documents
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # 1. Similarity Search (k=15). Run without native filter to retrieve compliance docs,
    # as we will perform research-specific sensitivity gating in python.
    results = db.similarity_search(query, k=15)
    
    formatted_results = []
    retrieved_doc_ids = set()
    for doc in results:
        doc_id = doc.metadata.get("doc_id", "Unknown")
        doc_type = doc.metadata.get("type", "Unknown")
        doc_sensitivity = doc.metadata.get("sensitivity", "Public")
        
        # Gating rule: only restrict 'Restricted' documents of type 'research' or starting with 'RN-'
        # from RMs without private access. Compliance policies (CMP) and Product Guides (PG) remain accessible.
        is_restricted_research = (doc_sensitivity == "Restricted") and (doc_type == "research" or doc_id.upper().startswith("RN-"))
        if not rm_access_to_private and is_restricted_research:
            continue
            
        retrieved_doc_ids.add(doc_id)
        formatted_results.append({
            "doc_id": doc_id,
            "type": doc_type,
            "date": doc.metadata.get("date", "Unknown"),
            "source": doc.metadata.get("source", "Unknown"),
            "sensitivity": doc_sensitivity,
            "content": doc.page_content
        })
        
    # 2. Dynamic keyword fallback rules from JSON configuration
    rules_path = os.path.join(os.path.dirname(__file__), "retriever_rules.json")
    keyword_map = {}
    implicit_rules = []
    if os.path.exists(rules_path):
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                keyword_map = config.get("keyword_map", {})
                implicit_rules = config.get("implicit_rules", [])
        except Exception as e:
            print(f"Error loading retriever_rules.json: {e}")
            
    q_lower = query.lower()
    expected_from_keywords = set()
    for kw, doc_id in keyword_map.items():
        if kw in q_lower:
            expected_from_keywords.add(doc_id)
            
    for rule in implicit_rules:
        if any(trigger in q_lower for trigger in rule.get("trigger_words", [])):
            for doc_id in rule.get("inject_doc_ids", []):
                expected_from_keywords.add(doc_id)
                
    # 3. Resolve fallback lookups for expected documents missing from initial search
    missing_docs = expected_from_keywords - retrieved_doc_ids
    for m_doc in missing_docs:
        # Direct lookup by metadata doc_id
        fallback_results = db.similarity_search(query, k=2, filter={"doc_id": m_doc})
        for doc in fallback_results:
            doc_id = doc.metadata.get("doc_id", "Unknown")
            doc_type = doc.metadata.get("type", "Unknown")
            doc_sensitivity = doc.metadata.get("sensitivity", "Public")
            
            # Gating rule check for fallback chunks
            is_restricted_research = (doc_sensitivity == "Restricted") and (doc_type == "research" or doc_id.upper().startswith("RN-"))
            if not rm_access_to_private and is_restricted_research:
                continue
                
            formatted_results.append({
                "doc_id": doc_id,
                "type": doc_type,
                "date": doc.metadata.get("date", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "sensitivity": doc_sensitivity,
                "content": doc.page_content
            })
            
    # 4. Deduplicate and group chunks by doc_id to ensure diversity and prevent truncation
    by_doc_id = {}
    for doc in formatted_results:
        d_id = doc["doc_id"]
        if d_id not in by_doc_id:
            by_doc_id[d_id] = []
        by_doc_id[d_id].append(doc)
        
    # Reassemble round-robin style: take 1st chunk of each doc, then 2nd, etc.
    diverse_results = []
    max_chunks_per_doc = 3
    for i in range(max_chunks_per_doc):
        for d_id, chunks in by_doc_id.items():
            if len(chunks) > i:
                diverse_results.append(chunks[i])
                
    # Return top 8 of the diverse chunks
    return diverse_results[:8]
