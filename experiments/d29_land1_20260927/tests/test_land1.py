"""Synthetic software fixtures only; no inference about real TN performance."""
from __future__ import annotations
import copy
import unittest
import numpy as np
import torch
from d29_platform.land1 import (run_land1, land1_adjoint, PlantBudgetInfeasible,
    MissingLand1Input, PROBABILITY_NAMES, hazard_to_probability, propagate_source_labels)
from d29_platform.land1_reference import torch_reference


def fixture(nt=9, nr=2, nl=3):
    rng = np.random.default_rng(1729)
    shape = (nt, nr, nl)
    initial = rng.uniform(2, 4, (nr, nl, 5))
    source = rng.uniform(.1, .4, (*shape, 4))
    out = rng.uniform(.01, .04, (*shape, 3))
    target = rng.uniform(3, 5, shape)
    levels = (.08, .015, .2, .03, .4, .1)
    probs = {n: np.full(shape, v) for n, v in zip(PROBABILITY_NAMES, levels)}
    m = np.broadcast_to(np.eye(nl) * .7 + np.ones((nl, nl)) * .3 / nl, (nr, nl, nl)).copy()
    return dict(initial=initial, sources=source, plant_target=target, plant_outflows=out,
                probabilities=probs, transitions={nt // 2: m})


class TestLand1(unittest.TestCase):
    def test_reference_and_full_adjoint(self):
        spec = fixture()
        actual = run_land1(**spec)
        rng = np.random.default_rng(33)
        gf = rng.normal(size=actual.fluxes.shape)
        gs = rng.normal(size=actual.states.shape)
        gfinal = rng.normal(size=actual.final.shape)
        tensor = lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)
        ref = {k: tensor(v) for k, v in spec.items() if k not in ("probabilities", "transitions")}
        ref["probabilities"] = {k: tensor(v) for k, v in spec["probabilities"].items()}
        ref["transitions"] = {k: tensor(v) for k, v in spec["transitions"].items()}
        flux, states = torch_reference(**ref)
        np.testing.assert_allclose(actual.fluxes, flux.detach().numpy(), atol=2e-14, rtol=1e-14)
        np.testing.assert_allclose(actual.states, states.detach().numpy(), atol=2e-14, rtol=1e-14)
        objective = (flux * torch.tensor(gf)).sum() + (states * torch.tensor(gs)).sum() + (states[-1] * torch.tensor(gfinal)).sum()
        objective.backward()
        grads = land1_adjoint(actual, grad_fluxes=gf, grad_states=gs, grad_final=gfinal)
        for name in ("initial", "sources", "plant_target", "plant_outflows"):
            np.testing.assert_allclose(grads[name], ref[name].grad.numpy(), atol=5e-13, rtol=2e-12, err_msg=name)
        for name in PROBABILITY_NAMES:
            np.testing.assert_allclose(grads["probabilities"][name], ref["probabilities"][name].grad.numpy(), atol=5e-13, rtol=2e-12, err_msg=name)
        for day in spec["transitions"]:
            np.testing.assert_allclose(grads["transitions"][day], ref["transitions"][day].grad.numpy(), atol=5e-13, rtol=2e-12)
        self.assertLess(actual.max_local_balance_kg, 1e-12)

    def test_multistep_difference(self):
        spec = fixture()
        result = run_land1(**spec)
        grad = land1_adjoint(result, grad_final=np.ones_like(result.final))
        for field, index in (("initial", (0, 0, 1)), ("sources", (2, 1, 1, 3)),
                             ("plant_target", (3, 0, 0)), ("plant_outflows", (4, 0, 0, 1))):
            for h in (1e-4, 1e-5, 1e-6):
                plus, minus = copy.deepcopy(spec), copy.deepcopy(spec)
                plus[field][index] += h
                minus[field][index] -= h
                fd = (run_land1(**plus).final.sum() - run_land1(**minus).final.sum()) / (2 * h)
                self.assertAlmostEqual(fd, grad[field][index], delta=2e-7)
        # Feasible transition perturbation preserves each old-land row sum.
        day = next(iter(spec["transitions"]))
        plus, minus = copy.deepcopy(spec), copy.deepcopy(spec)
        for obj, sign in ((plus, 1), (minus, -1)):
            obj["transitions"][day][0, 0, 0] += sign * 1e-5
            obj["transitions"][day][0, 0, 1] -= sign * 1e-5
        fd = (run_land1(**plus).final.sum() - run_land1(**minus).final.sum()) / 2e-5
        g = grad["transitions"][day]
        self.assertAlmostEqual(fd, g[0, 0, 0] - g[0, 0, 1], delta=2e-7)

    def test_streaming_causality_and_next_day_organic(self):
        spec = fixture()
        full = run_land1(**spec)
        stream = run_land1(**spec, keep_history=False)
        np.testing.assert_array_equal(full.final, stream.final)
        np.testing.assert_array_equal(full.fluxes, stream.fluxes)
        other = copy.deepcopy(spec)
        other["sources"][6:] *= 2
        future = run_land1(**other)
        np.testing.assert_array_equal(full.states[:7], future.states[:7])
        spec = fixture(nt=3, nr=1, nl=1)
        spec["initial"][:] = 0
        spec["sources"][:] = 0
        spec["plant_target"][:] = 0
        spec["plant_outflows"][:] = 0
        spec["transitions"] = {}
        spec["sources"][0, 0, 0, 1] = 1
        out = run_land1(**spec)
        self.assertEqual(out.fluxes[0].sum(), 0)
        self.assertGreater(out.fluxes[1, 0, 0, 0], 0)

    def test_missing_and_infeasible(self):
        spec = fixture()
        del spec["probabilities"]["available_loss"]
        with self.assertRaises(MissingLand1Input):
            run_land1(**spec)
        spec = fixture()
        spec["initial"][:] = 0
        spec["sources"][:] = 0
        with self.assertRaises(PlantBudgetInfeasible) as cm:
            run_land1(**spec)
        self.assertEqual(cm.exception.day, 0)
        self.assertGreater(cm.exception.shortfall_kg, 0)

    def test_zero_and_risk_cap(self):
        spec = fixture()
        spec["initial"][:] = 0
        spec["sources"][:] = 0
        spec["plant_target"][:] = 0
        spec["plant_outflows"][:] = 0
        result = run_land1(**spec)
        self.assertEqual(result.fluxes.sum() + result.final.sum(), 0)
        p, dp = hazard_to_probability(np.array([0., 1e-12, 700., 701.]))
        self.assertEqual(p[0], 0)
        self.assertEqual(dp[-1], 0)
        self.assertEqual(dp[-2], 0)

    def test_tags_and_land_transfer(self):
        spec = fixture()
        result = run_land1(**spec)
        initial = np.zeros((*spec["initial"].shape[:-1], 3, 5))
        source = np.zeros((*spec["sources"].shape[:-1], 3, 4))
        initial[..., 0, :] = spec["initial"]
        source[..., 1, :] = .3 * spec["sources"]
        source[..., 2, :] = .7 * spec["sources"]
        tags = propagate_source_labels(result, tagged_initial=initial, tagged_sources=source,
                                      labels=("initial", "source_a", "source_b"), record_history=True)
        self.assertLess(tags.max_local_balance_kg, 2e-13)
        self.assertLess(tags.max_state_sum_error_kg, 2e-13)
        self.assertLess(tags.max_flux_sum_error_kg, 2e-13)
        np.testing.assert_allclose(tags.state_history.sum(-2), result.states, atol=2e-13)
        np.testing.assert_allclose(tags.flux_history.sum(-2), result.fluxes, atol=2e-13)
        streaming = propagate_source_labels(result, tagged_initial=initial, tagged_sources=source,
                                            labels=tags.labels)
        np.testing.assert_array_equal(tags.final, streaming.final)
        self.assertIsNone(streaming.state_history)

    def test_checkpoint_resume_and_no_water(self):
        spec = fixture()
        full = run_land1(**spec)
        split = 4
        def part(start, stop, initial):
            return dict(initial=initial, sources=spec["sources"][start:stop],
                        plant_target=spec["plant_target"][start:stop],
                        plant_outflows=spec["plant_outflows"][start:stop],
                        probabilities={k: v[start:stop] for k, v in spec["probabilities"].items()},
                        transitions={d - start: m for d, m in spec["transitions"].items() if start <= d < stop})
        first = run_land1(**part(0, split, spec["initial"]))
        second = run_land1(**part(split, len(spec["sources"]), first.final))
        np.testing.assert_array_equal(full.final, second.final)
        np.testing.assert_array_equal(full.fluxes, np.concatenate((first.fluxes, second.fluxes)))
        dry = copy.deepcopy(spec)
        dry["probabilities"]["mobilize"][:] = 0
        dry["probabilities"]["lower_release"][:] = 0
        zero = run_land1(**dry)
        self.assertEqual(zero.fluxes[..., :2].sum(), 0)
        self.assertLess(zero.max_local_balance_kg, 1e-12)

    def test_uptake_kink_is_one_sided(self):
        spec = fixture(nt=1, nr=1, nl=1)
        spec["initial"][:] = 0
        spec["initial"][0, 0, 3] = 1
        spec["sources"][:] = 0
        spec["plant_target"][:] = 1
        spec["plant_outflows"][:] = 0
        spec["transitions"] = {}
        actual = run_land1(**spec)
        cotangent = np.zeros_like(actual.fluxes)
        cotangent[..., 0] = 1
        g = land1_adjoint(actual, grad_fluxes=cotangent)
        self.assertEqual(g["initial"][0, 0, 3], 0)
        for h in (1e-4, 1e-5, 1e-6):
            plus, minus = copy.deepcopy(spec), copy.deepcopy(spec)
            plus["initial"][0, 0, 3] += h
            minus["initial"][0, 0, 3] -= h
            right = (run_land1(**plus).fluxes[0, 0, 0, 0] - actual.fluxes[0, 0, 0, 0]) / h
            left = (actual.fluxes[0, 0, 0, 0] - run_land1(**minus).fluxes[0, 0, 0, 0]) / h
            self.assertAlmostEqual(left, 0, delta=1e-12)
            self.assertAlmostEqual(right, .2 * .4, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
