import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer import extract_citations, label_chunks, resolve_labels


def test_extract_citations():
    text = ("Supply risk [AAPL-10-K-2025-FY-risk_factors-5, TSLA-10-Q-2024-Q2-mdna-12]. "
            "Again [AAPL-10-K-2025-FY-risk_factors-5]. Bank [BRK-10-K-2024-FY-other-3] [JPM-10-K-2025-FY-x]")
    assert extract_citations(text) == ["AAPL-10-K-2025-FY-risk_factors-5", "TSLA-10-Q-2024-Q2-mdna-12",
                                       "BRK-10-K-2024-FY-other-3"]


def test_labels_resolve_to_ids_and_flag_unknown():
    results = {"LLY": [{"id": "LLY-10-K-2025-FY-risk_factors-18"}, {"id": "LLY-10-K-2025-FY-risk_factors-1"}],
               "MRK": [{"id": "MRK-10-K-2024-FY-risk_factors-14"}]}
    labels = label_chunks(results)
    assert labels["LLY-2"] == "LLY-10-K-2025-FY-risk_factors-1"
    text, bad = resolve_labels("Pricing [LLY-1, LLY-2]; privacy [MRK-1; MRK-9]. FY2025 [note] [Q2 FY2025]", labels)
    assert text == ("Pricing [LLY-10-K-2025-FY-risk_factors-18, LLY-10-K-2025-FY-risk_factors-1]; privacy "
                    "[MRK-10-K-2024-FY-risk_factors-14, MRK-9]. FY2025 [note] [Q2 FY2025]")
    assert bad == ["MRK-9"]
