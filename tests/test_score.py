import math

from test_train_conditional import LATENT_DIM, _DATA_ONLY, _make_case
from wrf_pinn.evaluation.score import HEIGHT_BANDS, score_cases
from wrf_pinn.models.conditional import ConditionalModel, ConditionalModelConfig
from wrf_pinn.training.train_conditional import ConditionalTrainConfig, train_conditional


def test_score_cases_covers_every_point():
    cases = [_make_case(f"c{i}", i) for i in range(4)]
    cfg = ConditionalTrainConfig(epochs=5, batch_cases=2, device="cpu", log_every=5, **_DATA_ONLY)
    model = ConditionalModel(LATENT_DIM, ConditionalModelConfig(hidden_layers=2, hidden_width=16))
    train_conditional(model, cases, cfg)

    out = score_cases(model, cases, cfg, read=lambda c: c, device="cpu", readers=2)
    assert len(out["by_height"]) == len(HEIGHT_BANDS) - 1
    assert sum(out["points_by_height"]) == sum(float(c.target_mask[:, 0].sum()) for c in cases)
    for name in ("r2", "rmse", "bias", "spread_pooled"):
        assert all(math.isfinite(v) for v in out["all"][name])
    assert set(out["losses"]) == {"data", "initial", "flow_boundary", "pde", "surface"}
