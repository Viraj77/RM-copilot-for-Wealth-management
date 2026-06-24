import sys
from pathlib import Path

# ====================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ====================================================

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

# ====================================================
# IMPORTS
# ====================================================

import json

from src.agent import agent

# ====================================================
# LOAD TEST CASES
# ====================================================

with open(
    "evaluation/test_dataset.json"
) as f:

    tests = json.load(f)

# ====================================================
# RUN TESTS
# ====================================================

for idx, test in enumerate(
    tests,
    start=1
):

    print("\n")
    print("=" * 80)

    print(f"TEST CASE {idx}")

    print("=" * 80)

    print("\nQUERY:")
    print(test["query"])

    print("\nEXPECTED INTENT:")
    print(test["intent"])

    print("\nRUNNING AGENT...\n")

    result = agent.invoke({

        "query": test["query"]
    })

    print(result["response"])

    print("\n")