"""Regression properties of the preserved density PDE discretization."""
import numpy as np
import pytest

from swarm_core import make_obstacle, make_initial, make_target, expert_flux, step_density, _face_flux_and_divergence


@pytest.mark.parametrize("preset", ["Нет", "Центральный блок", "Коридор", "Стена с проходом"])
def test_density_remains_finite_positive_and_outside_obstacles(preset):
    obstacle = make_obstacle(24, preset)
    rho = make_initial(24, "Слева", obstacle, 42)
    target = make_target(24, "Круг справа", obstacle, 42)
    for _ in range(20):
        jx, jy = expert_flux(rho, target, obstacle)
        # Closed boundaries and impermeable masks must not create net face flux.
        _, _, divergence = _face_flux_and_divergence(jx, jy, obstacle)
        assert abs(divergence.sum()) < 1e-10
        rho, *_ = step_density(rho, jx, jy, obstacle, .12)
        assert np.isfinite(rho).all() and (rho >= 0).all()
        assert abs(rho.sum() - 1) < 1e-12
        assert np.count_nonzero(rho[obstacle]) == 0


def test_original_post_mission_scenarios():
    # Original regression was a CLI main and otherwise escaped pytest discovery.
    from test_mission_scenarios import main
    main()
