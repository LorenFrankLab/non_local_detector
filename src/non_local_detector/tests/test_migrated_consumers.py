"""Execute shipped consumer calls without their recording files or databases."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from non_local_detector import Environment, time_edges_from_centers
from non_local_detector.analysis import distance1D
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOKS = [
    "notebooks/01_models_and_validation/sorted_spikes_detector_test.ipynb",
    "notebooks/02_likelihood_models/gradient_descent_optimization.ipynb",
    "notebooks/02_likelihood_models/place_field_fitting_real_data_v2.ipynb",
    "notebooks/02_likelihood_models/place_field_fitting_synthetic.ipynb",
    "notebooks/02_likelihood_models/weighted_place_fields.ipynb",
]


def _require_source(path):
    if not path.is_file():
        pytest.skip(
            "Repository notebook/script sources are absent from this installation"
        )
    return path


def _sources(path):
    _require_source(path)
    if path.suffix == ".ipynb":
        return [
            "".join(cell["source"])
            for cell in json.loads(path.read_text())["cells"]
            if cell["cell_type"] == "code"
        ]
    return [path.read_text()]


def _calls(path, names):
    for source in _sources(path):
        try:
            tree = ast.parse(source)
        except SyntaxError:  # IPython help/magics are not executed here.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in names:
                    yield node


def _evaluate_call(node, namespace):
    return eval(compile(ast.Expression(node), "shipped consumer", "eval"), namespace)


def _result_grid(start=0.0, stop=1.0, n_bins=500):
    edges = np.linspace(start, stop, n_bins + 1)
    return xr.Dataset(
        coords={
            "time": (edges[:-1] + edges[1:]) / 2,
            "time_bin_start": ("time", edges[:-1]),
            "time_bin_end": ("time", edges[1:]),
            "is_missing": ("time", np.zeros(n_bins, bool)),
        }
    )


def test_camera_tracking_alignment_preserves_perfect_500_bin_decode():
    from non_local_detector.analysis import align_tracking_to_results

    camera_time = np.arange(31) / 30
    tracking = pd.DataFrame(
        {"x": 100 * camera_time, "y": 0.0, "segment": 0, "angle": 0.0},
        index=camera_time,
    )
    results = _result_grid()
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x", "y"],
        valid_position_intervals=[[0.0, 1.0]],
        categorical_columns=["segment"],
        circular_columns=["angle"],
    )
    assert valid.all()
    np.testing.assert_array_equal(aligned.index, results.time)
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(100.0, 0.0))
    graph.add_edge(0, 1, distance=100.0)
    mental = np.column_stack([100 * results.time.values, np.zeros(500)])
    posterior = xr.DataArray(
        np.ones((500, 1)), dims=("time", "position"), coords={"time": results.time}
    )
    with patch.object(
        distance1D,
        "_get_MAP_estimate_2d_position_edges",
        return_value=(mental, np.tile([0, 1], (500, 1))),
    ):
        trajectory = distance1D.get_trajectory_data(
            posterior,
            graph,
            object(),
            aligned[["x", "y"]],
            aligned["segment"].astype(int),
            aligned["angle"],
        )
    distances = distance1D.get_ahead_behind_distance(graph, *trajectory)
    assert distances.shape == (500,)
    np.testing.assert_allclose(distances, 0.0, atol=1e-12)


def test_alignment_preserves_gaps_missingness_and_endpoint_support():
    from non_local_detector.analysis import align_tracking_to_results

    tracking = pd.DataFrame(
        {"x": [0.0, 1.0, np.nan, 3.0, 4.0, 100.0, 101.0]},
        index=[0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0],
    )
    results = _result_grid(-0.5, 11.5, 24)
    results["is_missing"][1] = True
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x"],
        valid_position_intervals=[[0.0, 4.0], [10.0, 11.0]],
    )
    expected = np.isin(np.arange(24), [2, 7, 8, 21, 22])
    np.testing.assert_array_equal(valid, expected)
    assert aligned.loc[~valid, "x"].isna().all()
    np.testing.assert_allclose(
        aligned.loc[valid, "x"], [0.75, 3.25, 3.75, 100.25, 100.75]
    )


def test_categorical_ownership_and_shortest_arc_stay_within_segments():
    from non_local_detector.analysis import align_tracking_to_results

    tracking = pd.DataFrame(
        {
            "x": [0.0, 1.0, 100.0, 101.0],
            "label": ["a", "b", "c", "d"],
            "angle": [3.1, -3.1, 0.1, 0.3],
        },
        index=[0.0, 1.0, 2.0, 3.0],
    )
    results = _result_grid(0, 3, 6)
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x"],
        valid_position_intervals=[[0.0, 1.0], [2.0, 3.0]],
        categorical_columns=["label"],
        circular_columns=["angle"],
    )
    np.testing.assert_array_equal(valid, [True, True, False, False, True, True])
    assert aligned.loc[valid, "label"].tolist() == ["a", "a", "c", "c"]
    assert (np.abs(aligned.loc[valid, "angle"].values[:2]) > 3.1).all()
    np.testing.assert_allclose(aligned.loc[valid, "x"], [0.25, 0.75, 100.25, 100.75])


@pytest.mark.parametrize(
    "name",
    NOTEBOOKS + ["notebooks/01_models_and_validation/sorted_spikes_detector_test.py"],
)
def test_shipped_notebook_likelihood_calls_execute(name):
    time = np.arange(100) * 0.002
    position = 5 + np.sin(np.arange(100) * 0.1)
    spikes = np.array([0.019, 0.057, 0.111, 0.153])
    env = Environment(place_bin_size=2, position_range=((0, 10),)).fit_place_grid(
        position[:, None], infer_track_interior=False
    )
    encoding = fit_sorted_spikes_kde_encoding_model(
        time, position, [spikes], env, disable_progress_bar=True
    )
    namespace = {
        "np": np,
        "time": time,
        "position": position,
        "env": env,
        "std": np.sqrt(12.5),
        "weights": np.ones(100),
        "sampling_frequency": 500,
        "out": encoding,
        "out1": encoding,
        "out2": encoding,
        "sample_edges": time_edges_from_centers(time),
        "times": spikes,
        "neuron_spike_times": spikes,
        "cell_ind": 0,
        "time_edges_from_centers": time_edges_from_centers,
        "fit_sorted_spikes_kde_encoding_model": fit_sorted_spikes_kde_encoding_model,
        "predict_sorted_spikes_kde_log_likelihood": predict_sorted_spikes_kde_log_likelihood,
        "get_spikecount_per_time_bin": get_spikecount_per_time_bin,
    }
    calls = list(
        _calls(
            ROOT / name,
            {
                "get_spikecount_per_time_bin",
                "fit_sorted_spikes_kde_encoding_model",
                "predict_sorted_spikes_kde_log_likelihood",
            },
        )
    )
    assert calls
    for node in calls:
        # The first weighted example uses a single spike array; later examples
        # use a population. Preserve each source expression's actual nesting.
        first_weighted = name.endswith("weighted_place_fields.ipynb") and (
            any(isinstance(arg, ast.List) for arg in node.args)
            or any(
                k.arg == "spike_times" and isinstance(k.value, ast.List)
                for k in node.keywords
            )
        )
        namespace["spike_times"] = spikes if first_weighted else [spikes]
        output = _evaluate_call(node, namespace)
        if isinstance(output, dict):
            assert output["rate_units"] == "Hz"
        else:
            assert output.shape[0] == len(time)
            assert np.isfinite(output).all()


@pytest.mark.parametrize(
    "filename", ["profile_memory.py", "profile_clusterless_kde.py"]
)
@pytest.mark.parametrize("implementation", ["reference", "log"])
def test_profiler_fit_and_prediction_calls_execute(filename, implementation):
    from non_local_detector.likelihoods import clusterless_kde, clusterless_kde_log

    backend = clusterless_kde if implementation == "reference" else clusterless_kde_log
    time = np.arange(100) * 0.002
    position = (5 + np.sin(np.arange(100) * 0.1))[:, None]
    spikes = [np.array([0.019, 0.057, 0.111, 0.153])]
    marks = [np.array([[2.0, 4.0], [5.0, 3.0], [7.0, 2.0], [4.0, 6.0]])]
    data = {
        "encoding": {
            "position_time": time,
            "position": position,
            "spike_times": spikes,
            "spike_waveform_features": marks,
        },
        "decoding": {
            "time": time,
            "position_time": time,
            "position": position,
            "spike_times": spikes,
            "spike_waveform_features": marks,
        },
        "environment": Environment(
            place_bin_size=2, position_range=((0, 10),)
        ).fit_place_grid(position, infer_track_interior=False),
        "params": {
            "sampling_frequency": 500,
            "position_std": np.sqrt(12.5),
            "waveform_std": 24.0,
            "block_size": 100,
        },
    }
    fit_name = (
        "fit_func"
        if filename == "profile_clusterless_kde.py"
        else f"fit_{implementation}"
    )
    predict_name = (
        "predict_func"
        if filename == "profile_clusterless_kde.py"
        else f"predict_{implementation}"
    )
    namespace = {
        "data": data,
        "np": np,
        "jnp": jnp,
        "time_edges_from_centers": time_edges_from_centers,
        fit_name: backend.fit_clusterless_kde_encoding_model,
        predict_name: backend.predict_clusterless_kde_log_likelihood,
    }
    fit_call = next(_calls(ROOT / "scripts" / filename, {fit_name}))
    namespace["encoding"] = _evaluate_call(fit_call, namespace)
    predict_call = next(_calls(ROOT / "scripts" / filename, {predict_name}))
    output = _evaluate_call(predict_call, namespace)
    assert output.shape == (len(time), 5)
    assert np.isfinite(output).all()


def test_real_data_notebook_covariate_plot_uses_aligned_rows():
    from non_local_detector.analysis import align_tracking_to_results

    path = ROOT / "notebooks/01_models_and_validation/model_checking_real_data.ipynb"
    source = next(
        s for s in _sources(path) if "def plot_posterior_consistency_vs_covariate(" in s
    )
    definition = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    )
    camera_time = np.arange(31) / 30
    results = _result_grid()
    aligned, valid = align_tracking_to_results(
        pd.DataFrame(
            {"position": camera_time, "head_speed": camera_time * 20}, index=camera_time
        ),
        results,
        position_columns=["position"],
        valid_position_intervals=[[0, 1]],
    )
    namespace = {
        "plt": plt,
        "np": np,
        "position_info": pd.DataFrame(
            {"head_speed": camera_time * 20, "linear_position": camera_time * 100},
            index=camera_time,
        ),
        "decode_position_info": aligned,
        "analysis_valid": valid,
        "cont_hpd_overlap": np.linspace(0, 1, 500),
        "cont_frag_hpd_overlap": np.linspace(1, 0, 500),
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    try:
        # Execute the notebook's actual example arguments, including the
        # tracking frame chosen by the consumer, rather than a rewritten call.
        for node in ast.parse(source).body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id == definition.name
                ):
                    _evaluate_call(node.value, namespace)
        scatter = plt.gcf().axes[0].collections[0]
        assert scatter.get_offsets().shape == (valid.sum(), 2)
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "cell_index,distance_name", [(42, "cont_dist"), (44, "con_frag_dist")]
)
def test_real_data_notebook_trajectory_and_distance_plot_use_actual_bins(
    cell_index, distance_name
):
    from non_local_detector.analysis import align_tracking_to_results

    path = _require_source(
        ROOT / "notebooks/01_models_and_validation/model_checking_real_data.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    camera_time = np.arange(31) / 30
    tracking = pd.DataFrame(
        {
            "head_position_x": camera_time * 100,
            "head_position_y": 0.0,
            "projected_x_position": camera_time * 100,
            "projected_y_position": 0.0,
            "head_speed": 100.0,
            "linear_position": camera_time * 100,
            "track_segment_id": 0,
            "head_orientation": 0.0,
        },
        index=camera_time,
    )
    results = _result_grid()
    results["is_missing"][42] = True
    state_bins = pd.MultiIndex.from_product(
        [["Continuous", "Fragmented"], [0.0]], names=["state", "position"]
    )
    results = results.assign_coords(
        xr.Coordinates.from_pandas_multiindex(state_bins, "state_bins")
    )
    results["acausal_posterior"] = (("time", "state_bins"), np.full((500, 2), 0.5))
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(100.0, 0.0))
    graph.add_edge(0, 1, distance=100.0)
    namespace = {
        "np": np,
        "pd": pd,
        "plt": plt,
        "align_tracking_to_results": align_tracking_to_results,
        "position_info": tracking,
        "time": camera_time,
        "time_edges": np.linspace(0, 1, 501),
        "valid_position_intervals": np.array([[0, 1]]),
        "cont_results": results,
        "cont_frag_results": results,
        "cont_model": object(),
        "cont_frag_model": object(),
        "track_graph": graph,
        "cont_hpd_overlap": np.linspace(0, 1, 500),
        "cont_frag_hpd_overlap": np.linspace(1, 0, 500),
    }
    # Execute the notebook's full alignment block with real result coordinates.
    alignment = "".join(cells[3]["source"]).split(
        "# Keep the measured tracking timeline separate from the decode clock."
    )[1]
    exec(compile(alignment, str(path), "exec"), namespace)
    valid = namespace["analysis_valid"]
    assert valid.sum() == 499
    mental = np.column_stack([100 * results.time.values[valid], np.zeros(valid.sum())])
    try:
        with patch.object(
            distance1D,
            "_get_MAP_estimate_2d_position_edges",
            return_value=(mental, np.tile([0, 1], (valid.sum(), 1))),
        ):
            exec(
                compile("".join(cells[cell_index]["source"]), str(path), "exec"),
                namespace,
            )
        distances = namespace[distance_name]
        assert distances.shape == (499,)
        np.testing.assert_allclose(distances, 0.0, atol=1e-12)
        exec(
            compile("".join(cells[cell_index + 1]["source"]), str(path), "exec"),
            namespace,
        )
        assert plt.gcf().axes[0].collections[0].get_offsets().shape == (499, 2)
    finally:
        plt.close("all")


def test_adjacent_tracking_intervals_do_not_interpolate_across_segment_boundaries():
    from non_local_detector.analysis import align_tracking_to_results

    tracking = pd.DataFrame({"x": [0.0, 1.0, 100.0, 101.0]}, index=[0.0, 0.9, 1.1, 2.0])
    results = _result_grid(0, 2, 20)
    aligned, valid = align_tracking_to_results(
        tracking,
        results,
        position_columns=["x"],
        valid_position_intervals=[[0, 1], [1, 2]],
    )
    assert not valid[9:11].any()
    assert aligned.iloc[9:11]["x"].isna().all()


def test_missing_covariates_do_not_interpolate_across_nan_runs():
    from non_local_detector.analysis import align_tracking_to_results

    tracking = pd.DataFrame(
        {"x": np.arange(5.0), "speed": [1.0, 1.0, np.nan, 3.0, 3.0]},
        index=np.arange(5.0),
    )
    aligned, valid = align_tracking_to_results(
        tracking,
        _result_grid(0, 4, 8),
        position_columns=["x"],
        valid_position_intervals=[[0, 4]],
    )
    assert valid.all()
    assert aligned.iloc[2:6]["speed"].isna().all()


def test_all_real_data_notebook_plot_examples_pass_decode_rows():
    def check_rows(time_slice, time, position, *args, **kwargs):
        assert len(time) == len(position) == 500

    namespace = {
        "plot_model_checking": check_rows,
        "plot_single_model_checking": check_rows,
        "time": np.arange(31) / 30,
        "position": np.arange(31),
        "decode_time": np.linspace(0.001, 0.999, 500),
        "decode_position": np.arange(500),
        "spike_times": [],
        "time_slice_ind": slice(10, 20),
        "ind": 20,
        "start_ext": 0,
        "end_ext": 40,
        "model_label": "Continuous",
        "other_label": "Fragmented",
        "color": "blue",
        "other_color": "orange",
    }
    for name in (
        "cont_model",
        "cont_frag_model",
        "model",
        "other_model",
        "cont_results",
        "cont_frag_results",
        "results",
        "other_results",
    ):
        namespace[name] = object()
    for name in (
        "cont_hpd_overlap",
        "cont_frag_hpd_overlap",
        "hpd_overlap",
        "other_hpd_overlap",
    ):
        namespace[name] = np.ones(500)
    path = ROOT / "notebooks/01_models_and_validation/model_checking_real_data.ipynb"
    calls = list(_calls(path, {"plot_model_checking", "plot_single_model_checking"}))
    assert len(calls) >= 10
    for call in calls:
        _evaluate_call(call, namespace)


def test_weighted_place_field_plot_uses_fitted_hz_without_rescaling():
    path = ROOT / "notebooks/02_likelihood_models/weighted_place_fields.ipynb"
    source = next(s for s in _sources(path) if "axes[1].fill_between(" in s)
    time = np.arange(100) * 0.002
    position = (5 + np.sin(np.arange(100) * 0.1))[:, None]
    env = Environment(place_bin_size=2, position_range=((0, 10),)).fit_place_grid(
        position, infer_track_interior=False
    )
    encoding = fit_sorted_spikes_kde_encoding_model(
        time,
        position,
        [np.array([0.019, 0.057, 0.111, 0.153])] * 11,
        env,
        disable_progress_bar=True,
    )
    fields = np.asarray(encoding["place_fields"])
    namespace = {
        "plt": plt,
        "env": env,
        "out1": {"place_fields": fields},
        "out2": {"place_fields": fields * 2},
        "sampling_frequency": 500,
    }
    try:
        exec(compile(source, str(path), "exec"), namespace)
        np.testing.assert_array_equal(
            namespace["axes"][0].lines[0].get_ydata(), fields[10]
        )
        np.testing.assert_array_equal(
            namespace["axes"][0].lines[1].get_ydata(), fields[10] * 2
        )
        assert namespace["axes"][0].get_ylabel() == "Rate (Hz)"
        assert namespace["axes"][1].get_ylabel() == "Difference (Hz)"
    finally:
        plt.close("all")


def test_migration_example_excludes_the_declared_shared_boundary_despite_rounding():
    from non_local_detector import calculate_time_edges

    path = _require_source(ROOT / "scripts/time_grid_migration_example.py")
    tree = ast.parse(path.read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "selected_spikes"
            for target in node.targets
        )
    )
    edges = calculate_time_edges([0.1, 0.3], 500)
    assert (
        edges[-1] > 0.3
    )  # A real floating-point construction, not a tolerance change.
    namespace = {
        "spike_times": [np.array([0.1, 0.2, 0.3])],
        "edges": edges,
        "stop": 0.3,
        "shared_stop": True,
    }
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    np.testing.assert_array_equal(namespace["selected_spikes"][0], [0.1, 0.2])


def test_spike_time_tutorial_plots_fitted_hz_without_rescaling():
    path = ROOT / "notebooks/03_state_transitions/spike_time_decoding_v2.ipynb"
    source = next(s for s in _sources(path) if "initial_place_fields =" in s)
    time = np.arange(100) * 0.002
    position = (5 + np.sin(np.arange(100) * 0.1))[:, None]
    env = Environment(place_bin_size=2, position_range=((0, 10),)).fit_place_grid(
        position, infer_track_interior=False
    )
    encoding = fit_sorted_spikes_kde_encoding_model(
        time,
        position,
        [np.array([0.019, 0.057, 0.111, 0.153])],
        env,
        disable_progress_bar=True,
    )
    fields = np.repeat(np.asarray(encoding["place_fields"]), 40, axis=0)
    namespace = {
        "plt": plt,
        "env": env,
        "detector_noEM": SimpleNamespace(
            encoding_model_={("", 0): {"place_fields": fields}}
        ),
        "detector": SimpleNamespace(
            encoding_model_={("", 0): {"place_fields": fields * 2}}
        ),
    }
    try:
        exec(compile(source, str(path), "exec"), namespace)
        first = namespace["axes"].flat[0]
        assert (
            first.collections[0].get_paths()[0].vertices[:, 1].max() == fields[0].max()
        )
        assert (
            first.collections[1].get_paths()[0].vertices[:, 1].max()
            == (fields[0] * 2).max()
        )
        assert first.get_ylabel() == "Rate (Hz)"
    finally:
        plt.close("all")
