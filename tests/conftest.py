import pytest

from patient_zero.config import GridConfig
from patient_zero.tasks import load_sample


@pytest.fixture
def tasks():
    return load_sample(3)


@pytest.fixture
def cfg(tmp_path):
    def make(**kw):
        base = dict(exp_id="t", dataset="sample", n_tasks=3, models=["fake"], out_dir=str(tmp_path), concurrency=2)
        return GridConfig(**{**base, **kw})

    return make
