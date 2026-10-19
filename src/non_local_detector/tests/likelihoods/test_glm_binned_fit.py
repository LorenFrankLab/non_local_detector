"""Sorted-spikes GLM fit on spatially binned sufficient statistics.

The Poisson log likelihood depends on position only through the basis, so the
fit sums spike counts and exposure within small spatial cells and evaluates the
basis at each cell's exposure-weighted mean position, instead of at every time
sample. These tests check the aggregation and that the binned fit is as close to
the true rates as the per-sample fit.
"""

import networkx as nx
import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_glm import (
    _aggregate_by_position_cell,
    fit_poisson_regressions,
    fit_sorted_spikes_glm_encoding_model,
    log_rate_prior_l2_penalty,
    make_spline_predict_matrix,
)


def test_aggregation_preserves_totals_and_weights_positions():
    rng = np.random.default_rng(0)
    position = rng.uniform(0.0, 10.0, size=(1000, 2))
    exposure = rng.uniform(0.0, 2.0, size=1000)
    counts = rng.poisson(1.0, size=(3, 1000)).astype(float)
    mean_position, cell_counts, cell_exposure = _aggregate_by_position_cell(
        position, counts, exposure, cell_size=2.5
    )
    np.testing.assert_allclose(cell_counts.sum(axis=1), counts.sum(axis=1))
    np.testing.assert_allclose(cell_exposure.sum(), exposure.sum())
    assert mean_position.shape == (cell_exposure.size, 2)
    # Each cell's position is the exposure-weighted mean of its samples.
    key = np.floor(position / 2.5)
    first = np.all(key == key[0], axis=1)
    expected = np.average(position[first], axis=0, weights=exposure[first])
    match = np.flatnonzero(np.all(np.floor(mean_position / 2.5) == key[0], axis=1))
    np.testing.assert_allclose(mean_position[match[0]], expected)


def test_aggregation_never_mixes_regions():
    """Samples in different regions (track-graph edges) never share a cell, even
    when their positions fall in the same cell."""
    position = np.array([[4.9], [5.1], [5.0], [5.05]])
    region = np.array([0, 1, 0, 1])
    mean_position, cell_counts, cell_exposure = _aggregate_by_position_cell(
        position, np.ones((1, 4)), np.ones(4), cell_size=10.0, region=region
    )
    assert cell_exposure.size == 2
    np.testing.assert_allclose(np.sort(mean_position.ravel()), [4.95, 5.075])


def test_binned_fit_is_as_close_to_the_truth_as_the_per_sample_fit():
    rng = np.random.default_rng(0)
    dt, n_time, span = 0.002, 150_000, 78.0
    walk = np.cumsum(rng.normal(0.0, 0.25, (n_time, 2)), axis=0)
    position = 1.0 + np.abs((walk % (2 * span)) - span)
    time = np.arange(n_time) * dt
    env = Environment(
        place_bin_size=3.5, position_range=((0, 80), (0, 80))
    ).fit_place_grid(position, infer_track_interior=True)
    centers = rng.uniform(5.0, 75.0, (12, 2))

    def rate(points):
        return np.stack(
            [
                0.3 + 15.0 * np.exp(-0.5 * np.sum((points - c) ** 2, 1) / 6.0**2)
                for c in centers
            ]
        )

    sample_counts = (rng.random((len(centers), n_time)) < rate(position) * dt).astype(
        float
    )
    spike_times = [time[c > 0] for c in sample_counts]
    enc = fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        spike_times,
        env,
        env.place_bin_edges_,
        env.edges_,
        env.is_track_interior_,
        env.is_track_boundary_,
        disable_progress_bar=True,
    )
    interior = env.is_track_interior_.ravel()
    bin_centers = env.place_bin_centers_[interior]
    predict = np.asarray(
        make_spline_predict_matrix(enc["emission_design_info"], bin_centers)
    )
    reference = np.asarray(
        fit_poisson_regressions(
            np.asarray(
                make_spline_predict_matrix(enc["emission_design_info"], position)
            ),
            sample_counts,
            np.full(n_time, dt),
            l2_penalty=log_rate_prior_l2_penalty(predict),
        )
    )
    occupancy = (
        np.histogram2d(
            position[:, 0], position[:, 1], bins=[env.edges_[0], env.edges_[1]]
        )[0].ravel()[interior]
        * dt
    )
    truth = rate(bin_centers).T

    def kl_per_spike(fitted):
        return np.sum(
            occupancy[:, None] * (truth * np.log(truth / fitted) - truth + fitted)
        ) / (sample_counts.sum())

    binned = np.asarray(enc["place_fields"]).T[interior]
    per_sample = np.exp(predict @ reference.T)
    assert kl_per_spike(binned) < kl_per_spike(per_sample) + 2e-3


def test_binned_track_graph_fit_does_not_mix_edges_at_a_zero_gap_seam():
    """With no gap between edges in linear position, a cell at the seam would mix
    the end of one edge with the start of the next; cells are per edge."""
    nodes = {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (70.0, 30.0), 3: (70.0, -30.0)}
    edges = [(0, 1), (1, 2), (1, 3)]
    graph = nx.Graph()
    for node, pos in nodes.items():
        graph.add_node(node, pos=pos)
    for eid, (u, v) in enumerate(edges):
        graph.add_edge(
            u,
            v,
            distance=float(np.linalg.norm(np.subtract(nodes[u], nodes[v]))),
            edge_id=eid,
        )
    env = Environment(
        place_bin_size=1.0, track_graph=graph, edge_order=edges, edge_spacing=0.0
    ).fit_place_grid()
    rng = np.random.default_rng(0)
    seq = (0, 1, 2, 1, 3, 1, 0)
    legs = [
        np.linspace(nodes[a], nodes[b], 300)
        for a, b in zip(seq[:-1], seq[1:], strict=True)
    ]
    position = np.tile(np.concatenate(legs), (12, 1))
    time = np.arange(position.shape[0]) / 100.0
    distance = np.linalg.norm(position - np.asarray(nodes[2]), axis=1)
    rate = 0.2 + 20.0 * np.exp(-0.5 * (distance / 4.0) ** 2)
    spikes = time[rng.random(time.size) < rate * 0.01]
    enc = fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        [spikes],
        env,
        env.place_bin_edges_,
        env.edges_,
        env.is_track_interior_,
        env.is_track_boundary_,
        disable_progress_bar=True,
    )
    basis = enc["emission_design_info"]
    arm3_start = basis.edge_starts[2]
    centers = env.place_bin_centers_.ravel()
    interior = env.is_track_interior_.ravel()
    field = np.asarray(enc["place_fields"])[0]
    assert (
        field[interior & (centers >= arm3_start) & (centers < arm3_start + 6.0)].max()
        < 0.5
    )


def _plus_maze(seed=0, n_laps=6):
    """Laps from the center out each arm of a plus-shaped maze in a 200 cm box
    (about a fifth of the box) and back, with lateral jitter, and place cells
    along the arms."""
    rng = np.random.default_rng(seed)
    center = np.array([100.0, 100.0])
    ends = [
        np.array([100.0, 0.0]),
        np.array([100.0, 200.0]),
        np.array([0.0, 100.0]),
        np.array([200.0, 100.0]),
    ]
    legs = []
    for _ in range(n_laps):
        for end in ends:
            out = np.linspace(center, end, 6000)
            legs += [out, out[::-1]]
    position = np.concatenate(legs)
    along_x = np.abs(position[:, 1] - 100.0) < 1e-9
    jitter = rng.uniform(-4.0, 4.0, len(position))
    position[:, 1] += np.where(along_x, jitter, 0.0)
    position[:, 0] += np.where(along_x, 0.0, jitter)
    n_time = len(position)
    time = np.arange(n_time) * 0.002
    env = Environment(place_bin_size=3.5).fit_place_grid(position)
    centers = np.array([[100, 20], [100, 180], [20, 100], [180, 100], [100, 100]])
    rate = np.stack(
        [
            0.3 + 15.0 * np.exp(-0.5 * np.sum((position - c) ** 2, 1) / 6.0**2)
            for c in centers
        ]
    )
    spike_times = [time[rng.random(n_time) < r * 0.002] for r in rate]
    return env, time, position, spike_times


def test_2d_basis_drops_columns_without_support_on_the_environment(monkeypatch):
    """Tensor-spline columns that are negligible on every interior bin are dropped:
    the fit has fewer coefficients and the same place fields as with every column."""
    from non_local_detector.likelihoods import sorted_spikes_glm

    env, time, position, spike_times = _plus_maze()
    args = (
        time,
        position,
        spike_times,
        env,
        env.place_bin_edges_,
        env.edges_,
        env.is_track_interior_,
        env.is_track_boundary_,
    )
    enc = fit_sorted_spikes_glm_encoding_model(*args, disable_progress_bar=True)
    monkeypatch.setattr(sorted_spikes_glm, "_GLM_COLUMN_SUPPORT_TOL", -1.0)
    every_column = fit_sorted_spikes_glm_encoding_model(
        *args, disable_progress_bar=True
    )

    interior = env.is_track_interior_.ravel()
    bin_centers = env.place_bin_centers_[interior]
    design = enc["emission_design_info"]
    full_predict = np.asarray(
        make_spline_predict_matrix(design.design_info, bin_centers)
    )
    assert np.asarray(enc["coefficients"]).shape[1] < 0.7 * full_predict.shape[1]
    assert np.asarray(every_column["coefficients"]).shape[1] == full_predict.shape[1]
    np.testing.assert_allclose(
        np.asarray(make_spline_predict_matrix(design, bin_centers)),
        full_predict[:, design.columns],
    )
    fitted = np.asarray(enc["place_fields"]).T[interior]
    full = np.asarray(every_column["place_fields"]).T[interior]
    assert np.median(np.abs(fitted - full) / full) < 1e-3


def _open_field_fit(bin_size, box, field_center, field_std, n_time):
    """A random walk in a ``box``-wide square with one Gaussian place field."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0.0, box / 400.0, (n_time, 2)), axis=0)
    position = np.abs((walk % (2 * box)) - box)
    time = np.arange(n_time) * 0.002
    env = Environment(
        place_bin_size=bin_size, position_range=((0, box), (0, box))
    ).fit_place_grid(position)
    distance = np.linalg.norm(position - np.asarray(field_center), axis=1)
    rate = 0.3 + 20.0 * np.exp(-0.5 * (distance / field_std) ** 2)
    spikes = [time[rng.random(n_time) < rate * 0.002]]
    args = (
        time,
        position,
        spikes,
        env,
        env.place_bin_edges_,
        env.edges_,
        env.is_track_interior_,
        env.is_track_boundary_,
    )
    return args, position, distance


def test_2d_column_support_includes_positions_between_bin_centers(monkeypatch):
    """With knots closer together than the bins, a column can be negligible at
    every bin center and still carry the field between them, where the fit and
    the local likelihood evaluate it. Such columns are kept: the fitted rates at
    the animal's positions match the fit with every column."""
    from non_local_detector.likelihoods import sorted_spikes_glm

    args, position, distance = _open_field_fit(
        bin_size=5.0,
        box=20.0,
        field_center=(11.3, 8.7),
        field_std=1.5,
        n_time=100_000,
    )

    def local_rates(enc):
        design = np.asarray(
            make_spline_predict_matrix(enc["emission_design_info"], position)
        )
        return np.exp(design @ np.asarray(enc["coefficients"])[0])

    pruned = local_rates(
        fit_sorted_spikes_glm_encoding_model(
            *args, emission_knot_spacing=1.0, disable_progress_bar=True
        )
    )
    monkeypatch.setattr(sorted_spikes_glm, "_GLM_COLUMN_SUPPORT_TOL", -1.0)
    every_column = local_rates(
        fit_sorted_spikes_glm_encoding_model(
            *args, emission_knot_spacing=1.0, disable_progress_bar=True
        )
    )
    in_field = distance < 1.0
    assert every_column[in_field].mean() > 10.0
    np.testing.assert_allclose(pruned[in_field], every_column[in_field], rtol=0.05)
