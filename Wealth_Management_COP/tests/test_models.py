"""
Unit tests for Pydantic data models — Phase 1 validation.
Run: pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

from src.models.client import (
    AssetClass,
    ClientProfile,
    EntitlementTier,
    PortfolioHolding,
    RiskProfile,
)
from src.models.brief import (
    Citation,
    ClientBrief,
    ComplianceStatus,
    Recommendation,
    SuitabilityVerdict,
)
from src.models.documents import (
    ENTITLEMENT_ACCESS,
    SENSITIVITY_LEVELS,
    DocType,
    DocumentChunk,
    RawDocument,
    RetrievedChunk,
    Sensitivity,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_holding():
    return PortfolioHolding(
        ticker="VTI",
        name="Vanguard Total Stock Market ETF",
        asset_class=AssetClass.EQUITY,
        allocation_pct=60.0,
        current_value=1_200_000.0,
        gain_loss_pct=12.3,
    )


@pytest.fixture
def sample_client(sample_holding):
    return ClientProfile(
        client_id="C-204",
        name="Sarah Chen",
        risk_profile=RiskProfile.BALANCED,
        investment_horizon="7-10 years",
        total_aum=2_000_000.0,
        holdings=[
            sample_holding,
            PortfolioHolding(
                ticker="AGG",
                name="iShares Aggregate Bond ETF",
                asset_class=AssetClass.FIXED_INCOME,
                allocation_pct=30.0,
                current_value=600_000.0,
                gain_loss_pct=-2.1,
            ),
            PortfolioHolding(
                ticker="CASH",
                name="Cash",
                asset_class=AssetClass.CASH,
                allocation_pct=10.0,
                current_value=200_000.0,
                gain_loss_pct=0.0,
            ),
        ],
    )


@pytest.fixture
def sample_citation():
    return Citation(
        doc_id="equity_product_guide",
        chunk_id="equity_product_guide_0",
        doc_type="product",
        source="equity_fund_product_guide.md",
        chunk_text="Equity funds are suitable for Balanced profiles with allocation up to 65%.",
        page_or_section="Section 2.1",
    )


@pytest.fixture
def sample_recommendation(sample_citation):
    return Recommendation(
        idea="Rebalance equity allocation from 60% to 55%",
        rationale="Current equity is at the upper bound for a balanced profile given elevated valuations",
        suitability=SuitabilityVerdict.SUITABLE,
        suitability_reasoning="Within balanced profile equity limit of 65%. Rebalancing reduces risk.",
        citations=[sample_citation],
        priority="medium",
    )


# ── ClientProfile Tests ───────────────────────────────────────────────────────

class TestClientProfile:
    def test_valid_client_id(self, sample_client):
        assert sample_client.client_id == "C-204"

    def test_invalid_client_id_format(self):
        with pytest.raises(ValidationError, match="pattern"):
            ClientProfile(
                client_id="204",  # Missing "C-" prefix
                name="Test",
                risk_profile=RiskProfile.BALANCED,
                investment_horizon="5 years",
                total_aum=100_000.0,
            )

    def test_allocation_sum_validation(self):
        """Holdings that don't sum to ~100% should raise ValidationError."""
        with pytest.raises(ValidationError, match="sum"):
            ClientProfile(
                client_id="C-001",
                name="Test Client",
                risk_profile=RiskProfile.BALANCED,
                investment_horizon="5 years",
                total_aum=500_000.0,
                holdings=[
                    PortfolioHolding(
                        ticker="VTI", name="VTI", asset_class=AssetClass.EQUITY,
                        allocation_pct=50.0, current_value=250_000.0, gain_loss_pct=5.0
                    ),
                    PortfolioHolding(
                        ticker="AGG", name="AGG", asset_class=AssetClass.FIXED_INCOME,
                        allocation_pct=30.0, current_value=150_000.0, gain_loss_pct=-1.0
                    ),
                    # Total = 80%, missing 20% — should fail
                ],
            )

    def test_allocation_by_asset_class(self, sample_client):
        breakdown = sample_client.allocation_by_asset_class
        assert breakdown["equity"] == pytest.approx(60.0)
        assert breakdown["fixed_income"] == pytest.approx(30.0)
        assert breakdown["cash"] == pytest.approx(10.0)

    def test_risk_profile_enum(self):
        for profile in ["conservative", "balanced", "growth", "aggressive"]:
            client = ClientProfile(
                client_id="C-001",
                name="Test",
                risk_profile=profile,
                investment_horizon="5 years",
                total_aum=100_000.0,
            )
            assert isinstance(client.risk_profile, RiskProfile)

    def test_negative_aum_rejected(self):
        with pytest.raises(ValidationError):
            ClientProfile(
                client_id="C-001",
                name="Test",
                risk_profile=RiskProfile.BALANCED,
                investment_horizon="5 years",
                total_aum=-1.0,
            )


# ── PortfolioHolding Tests ────────────────────────────────────────────────────

class TestPortfolioHolding:
    def test_allocation_bounds(self):
        with pytest.raises(ValidationError):
            PortfolioHolding(
                ticker="TEST", name="Test", asset_class=AssetClass.EQUITY,
                allocation_pct=101.0,  # Over 100%
                current_value=100.0, gain_loss_pct=0.0
            )

    def test_negative_current_value_rejected(self):
        with pytest.raises(ValidationError):
            PortfolioHolding(
                ticker="TEST", name="Test", asset_class=AssetClass.EQUITY,
                allocation_pct=50.0, current_value=-1.0, gain_loss_pct=0.0
            )


# ── ClientBrief Tests ─────────────────────────────────────────────────────────

class TestClientBrief:
    def test_valid_brief(self, sample_recommendation):
        brief = ClientBrief(
            client_id="C-204",
            client_name="Sarah Chen",
            risk_profile=RiskProfile.BALANCED,
            portfolio_summary="60% equity, 30% fixed income, 10% cash",
            portfolio_risk_assessment="Portfolio is aligned with balanced risk profile.",
            total_aum=2_400_000.0,
            recommendations=[sample_recommendation],
            compliance_status=ComplianceStatus.CLEARED,
            talking_points=["Portfolio is on track.", "Consider rebalancing equity."],
        )
        assert brief.compliance_status == ComplianceStatus.CLEARED
        assert not brief.escalation_required

    def test_blocked_auto_sets_escalation(self, sample_recommendation):
        brief = ClientBrief(
            client_id="C-204",
            client_name="Sarah Chen",
            risk_profile=RiskProfile.BALANCED,
            portfolio_summary="Test",
            portfolio_risk_assessment="Test",
            total_aum=2_400_000.0,
            recommendations=[sample_recommendation],
            compliance_status=ComplianceStatus.BLOCKED,
            compliance_notes="Recommendation requires licensed advice.",
            talking_points=["Escalated to licensed advisor."],
        )
        assert brief.escalation_required is True

    def test_talking_points_required(self):
        with pytest.raises(ValidationError):
            ClientBrief(
                client_id="C-204",
                client_name="Sarah Chen",
                risk_profile=RiskProfile.BALANCED,
                portfolio_summary="Test",
                portfolio_risk_assessment="Test",
                total_aum=2_400_000.0,
                compliance_status=ComplianceStatus.CLEARED,
                talking_points=[],  # Empty — should fail
            )


# ── Recommendation Tests ──────────────────────────────────────────────────────

class TestRecommendation:
    def test_citation_required(self, sample_citation):
        rec = Recommendation(
            idea="Rebalance",
            rationale="Risk alignment",
            suitability=SuitabilityVerdict.SUITABLE,
            suitability_reasoning="Within allocation limits.",
            citations=[sample_citation],
        )
        assert len(rec.citations) == 1

    def test_no_citations_rejected(self):
        with pytest.raises(ValidationError, match="too_short"):
            Recommendation(
                idea="Rebalance",
                rationale="Risk alignment",
                suitability=SuitabilityVerdict.SUITABLE,
                suitability_reasoning="Within limits.",
                citations=[],  # Must have ≥1 citation
            )

    def test_unsuitable_requires_detailed_reasoning(self, sample_citation):
        with pytest.raises(ValidationError, match="reasoning"):
            Recommendation(
                idea="Buy ARKK",
                rationale="High growth potential",
                suitability=SuitabilityVerdict.UNSUITABLE,
                suitability_reasoning="Bad.",  # Too short
                citations=[sample_citation],
            )


# ── DocumentChunk Tests ───────────────────────────────────────────────────────

class TestDocumentChunk:
    def test_chroma_metadata_serialization(self):
        chunk = DocumentChunk(
            doc_id="doc_001",
            chunk_id="doc_001_0",
            doc_type=DocType.POLICY,
            source="conservative_portfolio_policy.md",
            content="Conservative clients should not hold high yield bonds.",
            sensitivity=Sensitivity.INTERNAL,
            chunk_index=0,
            page_or_section="Section 2.1",
        )
        meta = chunk.to_chroma_metadata()
        assert meta["doc_type"] == "policy"
        assert meta["sensitivity"] == "internal"
        assert all(isinstance(v, str) for v in meta.values())

    def test_entitlement_access_map(self):
        assert ENTITLEMENT_ACCESS["standard"] == 0
        assert ENTITLEMENT_ACCESS["premium"] == 1
        assert ENTITLEMENT_ACCESS["institutional"] == 2

    def test_sensitivity_levels(self):
        assert SENSITIVITY_LEVELS[Sensitivity.PUBLIC] < SENSITIVITY_LEVELS[Sensitivity.INTERNAL]
        assert SENSITIVITY_LEVELS[Sensitivity.INTERNAL] < SENSITIVITY_LEVELS[Sensitivity.RESTRICTED]
