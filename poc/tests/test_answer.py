import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer import extract_citations


def test_extract_citations():
    text = ("Supply risk [AAPL-10-K-2025-FY-risk_factors-5, TSLA-10-Q-2024-Q2-mdna-12]. "
            "Again [AAPL-10-K-2025-FY-risk_factors-5]. Bank [BRK-10-K-2024-FY-other-3] [JPM-10-K-2025-FY-x]")
    assert extract_citations(text) == ["AAPL-10-K-2025-FY-risk_factors-5", "TSLA-10-Q-2024-Q2-mdna-12",
                                       "BRK-10-K-2024-FY-other-3"]
