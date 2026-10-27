"""The clusterless GMM fit validates its parameters and carries predict's tiling.

A detector passes ``clusterless_algorithm_params`` only to the encoding fit and
calls predict with the fitted encoding model, so predict's memory knobs reach it
only through the model the fit returns. Unknown fit parameters used to be
dropped silently, so a misspelled option fell back to its default.
"""

import numpy as np
import pytest

from non_local_detector import ClusterlessDecoder, time_edges_from_centers
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import clusterless_gmm
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data

pytestmark = pytest.mark.unit

N_DECODE = 300


@pytest.fixture(scope="module")
def sim():
    return make_simulated_run_data(
        n_tetrodes=3, place_field_means=np.arange(0, 90, 15), n_runs=1, seed=0
    )


def _fit(sim, **params):
    environment = Environment(place_bin_size=5.0).fit_place_grid(sim.position)
    return clusterless_gmm.fit_clusterless_gmm_encoding_model(
        sim.position_time,
        sim.position,
        sim.spike_times,
        sim.spike_waveform_features,
        environment,
        gmm_components_occupancy=4,
        gmm_components_joint=4,
        disable_progress_bar=True,
        **params,
    )


def test_unknown_fit_parameter_is_rejected(sim):
    with pytest.raises(ValidationError, match="gmm_componets_joint") as error:
        _fit(sim, gmm_componets_joint=99)
    # The message lists the valid names so the typo is easy to fix.
    assert "gmm_components_joint" in str(error.value)
    assert "bin_tile_size" in str(error.value)


def test_detector_default_parameters_are_accepted(sim):
    """A detector passes its default clusterless parameters to every algorithm."""
    model = _fit(sim, waveform_std=24.0, position_std=6.0, block_size=10_000)
    assert len(model["joint_models"]) == len(sim.spike_times)


def test_fit_stores_predict_tiling(sim):
    model = _fit(sim, spike_block_size=16, bin_tile_size=8)
    assert model["spike_block_size"] == 16
    assert model["bin_tile_size"] == 8

    default = _fit(sim)
    assert default["spike_block_size"] == 1000
    assert default["bin_tile_size"] is None


def test_detector_tiling_parameters_reach_predict(sim, monkeypatch):
    """``bin_tile_size`` / ``spike_block_size`` set on the detector tile predict,
    and tiling leaves the decoded posterior unchanged within float32 rounding."""
    spike_block_size, bin_tile_size = 16, 8
    tile_shapes = []
    accumulate = clusterless_gmm._accumulate_log_likelihood_block

    def record_tile(log_likelihood, joint_logp_tile, *args):
        tile_shapes.append(joint_logp_tile.shape)
        return accumulate(log_likelihood, joint_logp_tile, *args)

    time_edges = time_edges_from_centers(sim.position_time[:N_DECODE])

    def decode(params):
        decoder = ClusterlessDecoder(
            clusterless_algorithm="clusterless_gmm",
            clusterless_algorithm_params={
                "gmm_components_occupancy": 4,
                "gmm_components_joint": 4,
                **params,
            },
        ).fit(
            sim.position_time,
            sim.position,
            sim.spike_times,
            sim.spike_waveform_features,
        )
        results = decoder.predict(
            sim.spike_times, sim.spike_waveform_features, time_edges=time_edges
        )
        return np.asarray(results.acausal_posterior), decoder

    monkeypatch.setattr(
        clusterless_gmm, "_accumulate_log_likelihood_block", record_tile
    )
    untiled, _ = decode({})
    assert tile_shapes == [], "the default predict path was tiled"
    tiled, decoder = decode(
        {"spike_block_size": spike_block_size, "bin_tile_size": bin_tile_size}
    )

    (encoding_model,) = decoder.encoding_model_.values()
    n_bins = encoding_model["interior_place_bin_centers"].shape[0]
    assert n_bins > bin_tile_size
    assert tile_shapes, "the tiled predict path was not used"
    assert max(rows for rows, _ in tile_shapes) == spike_block_size
    assert max(columns for _, columns in tile_shapes) == bin_tile_size
    np.testing.assert_allclose(tiled, untiled, rtol=1e-5, atol=1e-6)
