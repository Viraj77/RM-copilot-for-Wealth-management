from datasets import Dataset

from ragas import evaluate


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# ====================================================
# SAMPLE DATASET
# ====================================================

data = {

    "question": [

        "Show best performing bond funds",

        "Compare equity and fixed income outlook",

        "Prepare quarterly review for client C-204"
    ],

    "answer": [

        """
        Investment-grade corporate bond funds
        and short-duration sovereign bond funds
        are performing well in the current
        elevated-rate environment.
        """,

        """
        Fixed income currently benefits from
        elevated yields while equities remain
        exposed to macroeconomic volatility.
        """,

        """
        Client portfolio remains diversified
        across equities and bonds with balanced
        risk exposure and moderate income stability.
        """
    ],

    "contexts": [

        [
            """
            Corporate bonds are benefiting
            from elevated rates and stable spreads.
            """
        ],

        [
            """
            Fixed income yields improved while
            equities remain volatile.
            """
        ],

        [
            """
            Client portfolio allocation includes
            diversified equity and bond exposure.
            """
        ]
    ],

    "ground_truth": [

        """
        Bond funds benefit from higher
        interest-rate environments.
        """,

        """
        Fixed income provides defensive
        characteristics during volatility.
        """,

        """
        Balanced portfolios contain diversified
        exposure across asset classes.
        """
    ]
}

# ====================================================
# CREATE DATASET
# ====================================================

dataset = Dataset.from_dict(data)

# ====================================================
# RUN EVALUATION
# ====================================================

result = evaluate(

    dataset=dataset,

    metrics=[

        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

# ====================================================
# PRINT RESULTS
# ====================================================

print("\n")
print("=" * 80)
print("RAGAS METRICS")
print("=" * 80)

print(result)