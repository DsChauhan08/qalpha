from quantum_alpha.backtesting.performance_gate import (
    aggregate_fundamentals,
    evaluate_gate,
)


def test_evaluate_gate_relaxes_required_count_when_benchmarks_sparse():
    metrics = {"total_return": 0.12, "cagr": 0.11}
    market = {"total_return": 0.05, "cagr": 0.04}
    quant = {"total_return": 0.07, "cagr": 0.06}

    gate = evaluate_gate(metrics, market, quant, min_ratio=0.9, min_metrics=50)

    assert gate.relaxed is True
    assert gate.required == 2
    assert gate.coverage == 2
    assert gate.passed is True


def test_evaluate_gate_uses_abs_lower_for_beta():
    metrics = {"beta": 0.18}
    market = {"beta": 1.0}
    quant = {"beta": 0.45}

    gate = evaluate_gate(metrics, market, quant, min_ratio=0.9, min_metrics=1)

    assert gate.coverage == 1
    assert gate.good_count == 1
    assert gate.passed is True


def test_aggregate_fundamentals_builds_derived_metrics():
    fundamentals = [
        {
            "pe_ratio": 20.0,
            "price_to_book": 3.0,
            "market_cap": 1_000_000_000.0,
            "free_cashflow": 70_000_000.0,
            "operating_cashflow": 120_000_000.0,
            "total_debt": 100_000_000.0,
            "total_cash": 20_000_000.0,
            "ebitda": 90_000_000.0,
            "return_on_equity": 0.15,
        },
        {
            "pe_ratio": 22.0,
            "price_to_book": 3.4,
            "market_cap": 1_200_000_000.0,
            "free_cashflow": 72_000_000.0,
            "operating_cashflow": 125_000_000.0,
            "total_debt": 110_000_000.0,
            "total_cash": 25_000_000.0,
            "ebitda": 95_000_000.0,
            "return_on_equity": 0.16,
        },
    ]

    out = aggregate_fundamentals(fundamentals)

    assert out["price_to_earnings"] == 21.0
    assert out["price_to_book"] == 3.2
    assert out["free_cashflow_yield"] is not None
    assert out["price_to_cashflow"] is not None
    assert out["net_debt_to_ebitda"] is not None
    assert out["roe"] == 0.155

