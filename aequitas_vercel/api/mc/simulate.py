import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_lib'))
from helpers import ok, err, body as get_body

def handler(request, response):
    if request.method == "OPTIONS":
        return ok({})
    try:
        from mc_engine import SimConfig, run_full_suite
        import numpy as np

        p = get_body(request)

        # Map frontend parameter names → SimConfig fields
        cfg = SimConfig(
            S0          = float(p.get("S0",    p.get("ps0", 175))),
            mu          = float(p.get("mu",    p.get("pmu", 0.12))),
            sigma       = float(p.get("sigma", p.get("psig", 0.25))),
            r           = float(p.get("r",     p.get("pr",   0.05))),
            q           = float(p.get("q",     p.get("pq",   0.0))),
            T           = float(p.get("T",     p.get("pT",   1.0))),
            n_sims      = min(int(float(p.get("n_sims", p.get("pn", 50))) * 1000), 50_000),
            K           = float(p.get("K",     p.get("pK",   180))),
            barrier     = float(p.get("barrier", p.get("pb", 140))),
            option_type = str(p.get("option_type", p.get("ot", "call"))),
            # Heston
            kappa = float(p.get("kappa", p.get("pkap",  2.0))),
            xi    = float(p.get("xi",    p.get("pxi",   0.3))),
            rho   = float(p.get("rho",   p.get("prho", -0.7))),
            # Jump-Diffusion
            lam   = float(p.get("lam",   p.get("plam",  0.75))),
            mu_j  = float(p.get("mu_j",  p.get("pmuj", -0.05))),
            sig_j = float(p.get("sig_j", p.get("psij",  0.10))),
            # SABR
            alpha = float(p.get("alpha", p.get("palp", 0.25))),
            beta       = float(p.get("beta",        p.get("pbet", 0.5))),
            nu         = float(p.get("nu",          p.get("pnu",  0.4))),
            # investment
            investment = float(p.get('investment', p.get('pinv', 10000))),
        )

        result = run_full_suite(cfg)
        # Wrap in job-queue compatible envelope for the polling frontend
        return ok({"job_id": "sync", "status": "done", "result": result})

    except Exception as e:
        import traceback
        return err(f"{e} | {traceback.format_exc().splitlines()[-1]}")
