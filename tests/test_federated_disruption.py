# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Federated Disruption
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Federated disruption prediction tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for federated learning framework (FedAvg / FedProx)."""

import json

import numpy as np
import pytest

from scpn_control.control.federated_disruption import (
    DifferentialPrivacyConfig,
    MACHINE_PROFILES,
    N_FEATURES,
    FederatedConfig,
    FederatedServer,
    PrivacyLedgerEntry,
    _init_mlp_weights,
    compose_privacy_epsilon,
    create_facility_clients_from_arrays,
    create_machine_clients,
    differential_privacy_clip,
    gaussian_mechanism_epsilon,
    run_synthetic_multifacility_benchmark,
)


def _make_clients(machines=("DIII-D", "JET", "KSTAR"), seed=42):
    cfgs = [{"machine": m, "n_train": 120, "n_test": 40} for m in machines]
    return create_machine_clients(cfgs, seed=seed)


# ── Factory & client basics ──────────────────────────────────────────


class TestMachineClients:
    def test_create_three_clients(self):
        clients = _make_clients()
        assert len(clients) == 3
        assert {c.machine for c in clients} == {"DIII-D", "JET", "KSTAR"}

    def test_data_shapes(self):
        clients = _make_clients()
        for c in clients:
            assert c.X_train.shape == (120, N_FEATURES)
            assert c.y_train.shape == (120,)
            assert c.X_test.shape == (40, N_FEATURES)
            assert set(np.unique(c.y_train)).issubset({0.0, 1.0})

    def test_get_data_size(self):
        clients = _make_clients()
        assert clients[0].get_data_size() == 120

    def test_unknown_machine_rejected(self):
        with pytest.raises(ValueError, match="Unknown machine"):
            create_machine_clients([{"machine": "TOKAMAK-X"}])


class TestFacilityArrayClients:
    def test_create_facility_clients_from_arrays_preserves_boundaries(self):
        clients = _make_clients(("DIII-D", "JET"), seed=11)
        datasets = {
            client.machine: {
                "X_train": client.X_train,
                "y_train": client.y_train,
                "X_test": client.X_test,
                "y_test": client.y_test,
            }
            for client in clients
        }
        restored = create_facility_clients_from_arrays(datasets, learning_rate=0.02)
        assert [client.machine for client in restored] == ["DIII-D", "JET"]
        assert restored[0].learning_rate == pytest.approx(0.02)

    def test_create_facility_clients_rejects_bad_feature_shape(self):
        with pytest.raises(ValueError, match="X_train"):
            create_facility_clients_from_arrays(
                {
                    "DIII-D": {
                        "X_train": np.zeros((4, N_FEATURES - 1)),
                        "y_train": np.zeros(4),
                        "X_test": np.zeros((2, N_FEATURES)),
                        "y_test": np.zeros(2),
                    }
                }
            )

    def test_create_facility_clients_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="binary disruption labels"):
            create_facility_clients_from_arrays(
                {
                    "DIII-D": {
                        "X_train": np.zeros((4, N_FEATURES)),
                        "y_train": np.array([0.0, 1.0, 0.5, 1.0]),
                        "X_test": np.zeros((2, N_FEATURES)),
                        "y_test": np.zeros(2),
                    }
                }
            )


# ── FedAvg aggregation ───────────────────────────────────────────────


class TestFedAvg:
    def test_weighted_average(self):
        rng = np.random.default_rng(0)
        w1 = _init_mlp_weights(rng)
        w2 = _init_mlp_weights(rng)
        updates = [
            {"weights": w1, "n_samples": 100},
            {"weights": w2, "n_samples": 300},
        ]
        cfg = FederatedConfig(machines=["DIII-D", "JET"])
        server = FederatedServer(cfg)
        avg = server.aggregate(updates)
        for key in w1:
            expected = w1[key] * 0.25 + w2[key] * 0.75
            np.testing.assert_allclose(avg[key], expected, atol=1e-12)

    def test_single_client_passthrough(self):
        rng = np.random.default_rng(1)
        w = _init_mlp_weights(rng)
        cfg = FederatedConfig(min_clients=1, machines=["DIII-D"])
        server = FederatedServer(cfg)
        avg = server.aggregate([{"weights": w, "n_samples": 50}])
        for key in w:
            np.testing.assert_allclose(avg[key], w[key], atol=1e-12)

    def test_empty_raises(self):
        cfg = FederatedConfig(machines=["DIII-D", "JET"])
        server = FederatedServer(cfg)
        with pytest.raises(ValueError, match="at least one"):
            server.aggregate([])

    def test_zero_sample_weight_rejected(self):
        rng = np.random.default_rng(1)
        w = _init_mlp_weights(rng)
        cfg = FederatedConfig(machines=["DIII-D", "JET"])
        server = FederatedServer(cfg)
        with pytest.raises(ValueError, match="positive client sample counts"):
            server.aggregate([{"weights": w, "n_samples": 0}])


# ── FedProx ──────────────────────────────────────────────────────────


class TestFedProx:
    def test_proximal_differs_from_plain(self):
        clients = _make_clients(("DIII-D", "JET"), seed=7)
        cfg_avg = FederatedConfig(n_rounds=3, local_epochs=3, machines=["DIII-D", "JET"], aggregation="fedavg")
        cfg_prox = FederatedConfig(
            n_rounds=3,
            local_epochs=3,
            machines=["DIII-D", "JET"],
            aggregation="fedprox",
            mu_proximal=0.5,
        )
        s_avg = FederatedServer(cfg_avg, seed=0)
        s_prox = FederatedServer(cfg_prox, seed=0)
        s_avg.train(clients, 3)
        s_prox.train(clients, 3)
        diffs = [
            float(np.max(np.abs(s_avg.global_weights[k] - s_prox.global_weights[k]))) for k in s_avg.global_weights
        ]
        assert max(diffs) > 1e-6


# ── Single round ─────────────────────────────────────────────────────


class TestRunRound:
    def test_round_returns_finite_metrics(self):
        clients = _make_clients()
        cfg = FederatedConfig(machines=["DIII-D", "JET", "KSTAR"])
        server = FederatedServer(cfg)
        result = server.run_round(clients)
        for m in result["client_metrics"]:
            assert 0.0 <= m["accuracy"] <= 1.0
            assert np.isfinite(m["loss"])
            assert m["n_samples"] > 0
            assert m["machine"] in MACHINE_PROFILES

    def test_too_few_clients_rejected(self):
        clients = _make_clients(("DIII-D",))
        cfg = FederatedConfig(min_clients=2, machines=["DIII-D"])
        server = FederatedServer(cfg)
        with pytest.raises(ValueError, match="Need >="):
            server.run_round(clients)

    def test_round_metrics_contain_required_keys(self):
        clients = _make_clients(("DIII-D", "JET"))
        cfg = FederatedConfig(min_clients=2, machines=["DIII-D", "JET"])
        server = FederatedServer(cfg)
        result = server.run_round(clients)
        for m in result["client_metrics"]:
            for key in ("accuracy", "precision", "recall", "f1", "loss", "n_samples"):
                assert key in m


# ── Full training ────────────────────────────────────────────────────


class TestFullTraining:
    def test_loss_decreases_over_rounds(self):
        clients = _make_clients(seed=99)
        cfg = FederatedConfig(
            n_rounds=8,
            local_epochs=5,
            learning_rate=0.02,
            machines=["DIII-D", "JET", "KSTAR"],
        )
        server = FederatedServer(cfg, seed=99)
        history = server.train(clients, 8)
        assert len(history) == 8
        first_loss = history[0]["mean_loss"]
        last_loss = history[-1]["mean_loss"]
        assert last_loss < first_loss

    def test_federation_helps_generalisation(self):
        """Global model accuracy >= mean of single-machine accuracies."""
        rng = np.random.default_rng(42)
        machines = ["DIII-D", "JET", "KSTAR"]
        clients = _make_clients(machines, seed=42)

        cfg = FederatedConfig(
            n_rounds=10,
            local_epochs=5,
            learning_rate=0.02,
            machines=machines,
        )
        server = FederatedServer(cfg, seed=42)
        server.train(clients, 10)

        # Evaluate global model on each client's test set
        global_accs = [c.local_evaluate(server.global_weights)["accuracy"] for c in clients]

        # Compare against locally-only trained models
        local_accs = []
        for c in clients:
            local_w = _init_mlp_weights(rng)
            local_w = c.local_train(local_w, 50)
            local_accs.append(c.local_evaluate(local_w)["accuracy"])

        assert np.mean(global_accs) >= np.mean(local_accs) - 0.05


# ── Differential privacy ─────────────────────────────────────────────


class TestDifferentialPrivacy:
    def test_clipping_bounds_norm(self):
        rng = np.random.default_rng(0)
        grads = _init_mlp_weights(rng)
        # Scale up to ensure clipping activates
        big = {k: v * 100.0 for k, v in grads.items()}
        clipped = differential_privacy_clip(big, max_norm=1.0, noise_sigma=0.0, rng=rng)
        total_norm = np.sqrt(sum(float(np.sum(g**2)) for g in clipped.values()))
        assert total_norm <= 1.0 + 1e-6

    def test_noise_adds_variance(self):
        rng = np.random.default_rng(1)
        grads = {"w": np.zeros((4, 4)), "b": np.zeros(4)}
        noised = differential_privacy_clip(grads, max_norm=10.0, noise_sigma=1.0, rng=rng)
        assert float(np.std(noised["w"])) > 0.1

    def test_gaussian_accountant_composes_linearly(self):
        epsilon = gaussian_mechanism_epsilon(2.0, 1e-5)
        assert epsilon > 0.0
        assert compose_privacy_epsilon(2.0, 1e-5, 3) == pytest.approx(3.0 * epsilon)

    def test_dp_round_records_privacy_ledger(self):
        clients = _make_clients(("DIII-D", "JET"), seed=123)
        dp_config = DifferentialPrivacyConfig(
            max_update_norm=0.05,
            noise_multiplier=1.5,
            delta=1e-5,
            seed=9,
        )
        cfg = FederatedConfig(
            n_rounds=2,
            local_epochs=1,
            machines=["DIII-D", "JET"],
            dp_config=dp_config,
        )
        server = FederatedServer(cfg, seed=3)
        result = server.run_round(clients)
        assert isinstance(result["privacy"], PrivacyLedgerEntry)
        assert result["privacy"].participating_clients == 2
        assert result["privacy"].delta == pytest.approx(1e-5)
        assert server.privacy_summary()["rounds"] == 1
        assert server.privacy_summary()["epsilon"] == pytest.approx(result["privacy"].epsilon_spent)


# ── Data heterogeneity convergence ───────────────────────────────────


class TestHeterogeneity:
    def test_skewed_distributions_converge(self):
        """Clients with different disruption fractions still converge."""
        cfgs = [
            {"machine": "DIII-D", "n_train": 100, "disruption_fraction": 0.2},
            {"machine": "JET", "n_train": 100, "disruption_fraction": 0.7},
            {"machine": "KSTAR", "n_train": 100, "disruption_fraction": 0.5},
        ]
        clients = create_machine_clients(cfgs, seed=77)
        cfg = FederatedConfig(n_rounds=8, local_epochs=5, machines=["DIII-D", "JET", "KSTAR"])
        server = FederatedServer(cfg, seed=77)
        history = server.train(clients, 8)
        assert history[-1]["mean_loss"] < history[0]["mean_loss"]


# ── Serialisation ────────────────────────────────────────────────────


class TestSerialisation:
    def test_round_trip(self):
        cfg = FederatedConfig(machines=["DIII-D", "JET"])
        server = FederatedServer(cfg, seed=5)
        state = server.get_state()
        blob = json.dumps(state)
        restored = FederatedServer.from_state(json.loads(blob))
        for key in server.global_weights:
            np.testing.assert_allclose(
                restored.global_weights[key],
                server.global_weights[key],
                atol=1e-12,
            )
        assert restored.config.aggregation == "fedavg"

    def test_dp_state_round_trip_preserves_privacy_ledger(self):
        clients = _make_clients(("DIII-D", "JET"), seed=33)
        cfg = FederatedConfig(
            machines=["DIII-D", "JET"],
            dp_config=DifferentialPrivacyConfig(max_update_norm=0.05, noise_multiplier=2.0, delta=1e-5, seed=4),
        )
        server = FederatedServer(cfg, seed=5)
        server.run_round(clients)
        restored = FederatedServer.from_state(json.loads(json.dumps(server.get_state())))
        assert restored.config.dp_config is not None
        assert restored.privacy_summary()["epsilon"] == pytest.approx(server.privacy_summary()["epsilon"])
        assert restored.privacy_ledger[0].participating_clients == 2


class TestSyntheticMultiFacilityBenchmark:
    def test_benchmark_reports_bounded_synthetic_evidence(self):
        summary = run_synthetic_multifacility_benchmark(
            machines=("DIII-D", "JET", "KSTAR"),
            n_rounds=2,
            local_epochs=1,
            dp_config=DifferentialPrivacyConfig(max_update_norm=0.1, noise_multiplier=2.0, delta=1e-5, seed=1),
        )
        assert summary.evidence_kind == "synthetic_multi_facility"
        assert summary.machines == ("DIII-D", "JET", "KSTAR")
        assert 0.0 <= summary.mean_accuracy <= 1.0
        assert np.isfinite(summary.mean_loss)
        assert summary.privacy_epsilon == pytest.approx(compose_privacy_epsilon(2.0, 1e-5, 2))


class TestConfigValidation:
    """Lines 244-257: FederatedConfig.__post_init__ raises on invalid params."""

    def test_invalid_n_rounds(self):
        with pytest.raises(ValueError, match="n_rounds"):
            FederatedConfig(n_rounds=0, machines=["DIII-D", "JET"])

    def test_invalid_local_epochs(self):
        with pytest.raises(ValueError, match="local_epochs"):
            FederatedConfig(local_epochs=0, machines=["DIII-D", "JET"])

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            FederatedConfig(learning_rate=-0.01, machines=["DIII-D", "JET"])

    def test_invalid_aggregation(self):
        with pytest.raises(ValueError, match="aggregation"):
            FederatedConfig(aggregation="invalid", machines=["DIII-D", "JET"])

    def test_invalid_mu_proximal(self):
        with pytest.raises(ValueError, match="mu_proximal"):
            FederatedConfig(mu_proximal=-1.0, machines=["DIII-D", "JET"])

    def test_invalid_min_clients(self):
        with pytest.raises(ValueError, match="min_clients"):
            FederatedConfig(min_clients=0, machines=["DIII-D", "JET"])

    def test_invalid_machine_name(self):
        with pytest.raises(ValueError, match="Unknown machine"):
            FederatedConfig(machines=["DIII-D", "NONEXISTENT"])


class TestMachineClientValidation:
    """Line 276: MachineClient rejects unknown machine names."""

    def test_unknown_machine_raises(self):
        from scpn_control.control.federated_disruption import MachineClient

        with pytest.raises(ValueError, match="Unknown machine"):
            MachineClient("TOKAMAK_X", np.zeros((10, 8)), np.zeros(10), np.zeros((5, 8)), np.zeros(5))
