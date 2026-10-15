import numpy as np
from swarm_core import SimConfig, run_simulation, make_obstacle, make_initial, make_target, expert_flux, step_density


def test_mass_and_mae_expert():
    n=48
    obs=make_obstacle(n,"Нет")
    rho=make_initial(n,"Слева",obs,1)
    target=make_target(n,"Круг справа",obs,1)
    mae0=np.mean(np.abs(rho-target))
    for _ in range(25):
        jx,jy=expert_flux(rho,target,obs)
        rho,_,_,_=step_density(rho,jx,jy,obs,0.12,0.0,0.0)
    assert abs(rho.sum()-1.0)<1e-10
    assert np.mean(np.abs(rho-target)) < mae0

if __name__=='__main__':
    test_mass_and_mae_expert()
    print('OK')
