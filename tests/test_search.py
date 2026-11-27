import json

import pytest

from src.data import load_config
from src.search import KINDS, focal_grid, focal_variant, sample_trials

VOCAB = {"a_idx": 10, "b_idx": 300}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("kind", KINDS)
def test_configuraciones_reproducibles_y_prefijo(cfg, kind):
    a, b = sample_trials(kind, cfg, 10), sample_trials(kind, cfg, 20)
    assert a == b[:10] and len(b) == 20


@pytest.mark.parametrize("kind", KINDS)
def test_configuraciones_distintas(cfg, kind):
    ts = sample_trials(kind, cfg, 20)
    claves = {json.dumps({k: v for k, v in t.items() if k != "id"}, sort_keys=True) for t in ts}
    assert len(claves) == 20


def test_rejilla_focal_y_nombres(cfg):
    g = focal_grid(cfg)
    assert len(g) == len(cfg["nn"]["search"]["focal"]["gamma"]) * len(cfg["nn"]["search"]["focal"]["alpha"])
    assert len({focal_variant("cw_03", f) for f in g}) == len(g)


def test_todas_las_configuraciones_se_construyen(cfg):
    keras = pytest.importorskip("keras")
    from src.models import build_model
    for kind in KINDS:
        for t in sample_trials(kind, cfg, 20):
            keras.backend.clear_session()
            m = build_model(kind, cfg, 20, VOCAB, **t["arch"])
            assert m.output_shape[-1] == 1