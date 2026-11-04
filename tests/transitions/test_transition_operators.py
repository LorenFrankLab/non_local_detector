"""Exact structured products against the unchanged dense model constructor."""

import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    TransitionContext,
    Uniform,
)
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.models._defaults import _ModelDefaults
from non_local_detector.models.base import _DetectorBase
from non_local_detector.observation_models import ObservationModel
from non_local_detector.transition_operators import (
    DEFAULT_MAX_DENSE_BYTES,
    DenseTransitionBudgetError,
    LazyDenseTransition,
    UnsupportedTransitionError,
    build_transition_operator,
)
from tests.conftest import precision_mode


def environment(shape=(5, 7), name="", holes=True):
    env = Environment(environment_name=name)
    axes = [np.arange(n, dtype=float) for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    env.place_bin_centers_ = np.stack([x.ravel() for x in mesh], axis=1)
    env.centers_shape_ = shape
    env.is_track_interior_ = np.ones(shape, dtype=bool)
    if holes and np.prod(shape) > 6:
        env.is_track_interior_.ravel()[[0, 3, -2]] = False
    return env


@pytest.mark.unit
@pytest.mark.parametrize("direction", [None, "outward"])
def test_auto_manifold_fallback_preserves_deferred_topology(direction):
    kwargs = {"place_bin_size": 1.0, "position_range": ((0.0, 5.0), (0.0, 6.0))}
    position = np.array([[0.0, 0.0], [5.0, 6.0]])
    eager = Environment(**kwargs).fit_place_grid(position, infer_track_interior=False)
    mask = np.ones(eager.centers_shape_, dtype=bool)
    mask[2, 1:-1] = False
    eager = Environment(**kwargs, is_track_interior=mask).fit_place_grid(
        position, infer_track_interior=False
    )
    lazy = Environment(**kwargs, is_track_interior=mask).fit_place_grid(
        position, infer_track_interior=False, compute_all_pairs_distances=False
    )
    count = len(eager.place_bin_centers_)
    reference = RandomWalk(
        use_manifold_distance=True, direction=direction
    ).make_state_transition([eager])
    descriptor = RandomWalk(use_manifold_distance=True, direction=direction)
    operator = build_transition_operator(
        [[descriptor]], [lazy], (count,), allow_dense_fallback=True
    )
    np.testing.assert_allclose(
        np.asarray(LazyDenseTransition(operator)), reference, rtol=1e-12, atol=1e-14
    )
    # The descriptor keeps only its configuration, not the dense view.
    assert vars(descriptor) == vars(
        RandomWalk(use_manifold_distance=True, direction=direction)
    )
    assert isinstance(lazy.distance_between_nodes_, LazyGraphDistances)


@pytest.mark.unit
@pytest.mark.parametrize("direction", [None, "outward"])
def test_dense_manifold_random_walk_uses_deferred_graph_distances(direction):
    """Deferred distances give the graph-distance walk, never the Euclidean one."""
    kwargs = {"place_bin_size": 1.0, "position_range": ((0.0, 5.0), (0.0, 6.0))}
    position = np.array([[0.0, 0.0], [5.0, 6.0]])
    shape = (
        Environment(**kwargs)
        .fit_place_grid(position, infer_track_interior=False)
        .centers_shape_
    )
    mask = np.ones(shape, dtype=bool)
    mask[2, 1:-1] = False
    environments = [
        Environment(**kwargs, is_track_interior=mask).fit_place_grid(
            position,
            infer_track_interior=False,
            compute_all_pairs_distances=compute,
        )
        for compute in (True, False)
    ]
    eager, lazy = (
        RandomWalk(
            use_manifold_distance=True, direction=direction
        ).make_state_transition([environment])
        for environment in environments
    )
    euclidean = RandomWalk(direction=None).make_state_transition([environments[0]])
    assert isinstance(environments[1].distance_between_nodes_, LazyGraphDistances)
    np.testing.assert_allclose(lazy, eager, rtol=1e-12, atol=1e-14)
    assert not np.allclose(lazy, euclidean)


@pytest.mark.unit
def test_manifold_fallback_cumulative_distance_and_block_budget_preflight(monkeypatch):
    position = np.array([[0.0, 0.0], [3.0, 4.0]])
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False, compute_all_pairs_distances=False
    )
    count = len(env.place_bin_centers_)
    calls = []
    monkeypatch.setattr(
        LazyGraphDistances, "to_dense", lambda *a, **k: calls.append(True)
    )
    with pytest.raises(DenseTransitionBudgetError, match="distance"):
        build_transition_operator(
            [[RandomWalk(use_manifold_distance=True)]],
            [env],
            (count,),
            allow_dense_fallback=True,
            max_dense_bytes=count * count * 8,
        )
    assert calls == []


@pytest.mark.unit
def test_invalid_scalar_gaussian_shape_is_rejected_before_dense_allocation(monkeypatch):
    env = environment((3, 4))

    def forbidden(*args, **kwargs):
        msg = "allocated an invalid spatial dense block"
        raise AssertionError(msg)

    monkeypatch.setattr(RandomWalk, "make_state_transition", forbidden)
    with pytest.raises(ValueError, match="shape"):
        build_transition_operator(
            [[RandomWalk(movement_var=np.array([[2.0, 0.2], [0.2, 2.0]]))]],
            [env],
            (1,),
            allow_dense_fallback=True,
        )


@pytest.mark.unit
def test_opaque_custom_fallback_with_deferred_distances_has_explicit_limit():
    class Custom:
        environment_name = ""

        def make_state_transition(self, environments, context=None):
            msg = "opaque constructor cannot reinterpret sparse distances"
            raise AssertionError(msg)

    position = np.array([[0.0, 0.0], [3.0, 4.0]])
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False, compute_all_pairs_distances=False
    )
    count = len(env.place_bin_centers_)
    with pytest.raises(UnsupportedTransitionError, match="Opaque custom"):
        build_transition_operator(
            [[Custom()]], [env], (count,), allow_dense_fallback=True
        )


@pytest.mark.unit
def test_slotted_custom_transition_preserves_explicit_eager_dense_fallback():
    class Custom:
        __slots__ = ()
        environment_name = ""

        def make_state_transition(self, environments, context=None):
            return np.eye(len(environments[0].place_bin_centers_))

    env = environment((3, 4))
    operator = build_transition_operator(
        [[Custom()]], [env], (12,), allow_dense_fallback=True
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), np.eye(12))


@pytest.mark.unit
def test_slotted_custom_never_receives_a_temporary_dense_environment():
    class Custom:
        __slots__ = ("environment",)
        environment_name = ""

        def make_state_transition(self, environments, context=None):
            self.environment = environments[0]
            msg = "opaque lazy fallback must reject before construction"
            raise AssertionError(msg)

    position = np.array([[0.0, 0.0], [3.0, 4.0]])
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False, compute_all_pairs_distances=False
    )
    descriptor = Custom()
    with pytest.raises(UnsupportedTransitionError, match="Opaque custom"):
        build_transition_operator(
            [[descriptor]],
            [env],
            (len(env.place_bin_centers_),),
            allow_dense_fallback=True,
        )
    assert not hasattr(descriptor, "environment")
    assert isinstance(env.distance_between_nodes_, LazyGraphDistances)


@pytest.mark.unit
def test_gaussian_forward_large_finite_input_with_disjoint_axis_mask():
    env = environment((2, 2), holes=False)
    env.is_track_interior_ = np.array([[False, True], [True, False]])
    transition = RandomWalk(movement_mean=15.0, movement_var=1.0)
    dense = transition.make_state_transition([env])
    operator = build_transition_operator([[transition]], [env], (4,))
    values = jnp.asarray([0.0, 1e38, 0.0, 0.0], dtype=jnp.float32)
    expected = values @ jnp.asarray(dense, dtype=values.dtype)
    actual = operator.forward(values)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


@pytest.mark.unit
def test_gaussian_backward_tiny_finite_input_with_disjoint_axis_mask():
    env = environment((2, 2), holes=False)
    env.is_track_interior_ = np.array([[False, True], [True, False]])
    transition = RandomWalk(movement_mean=15.0, movement_var=1.0)
    dense = transition.make_state_transition([env])
    operator = build_transition_operator([[transition]], [env], (4,))
    values = jnp.full(4, 1e-35, dtype=jnp.float32)
    expected = jnp.asarray(dense, dtype=values.dtype) @ values
    np.testing.assert_allclose(operator.backward(values), expected, rtol=1e-6, atol=0)


@pytest.mark.unit
def test_scaled_subnormal_valid_terms_reject_before_their_row_sum_underflows():
    # All original nearest valid PDFs are normal float64. The projected axis
    # maxima lie in holes, leaving individually subnormal float32 terms even
    # though their summed scaled row normalizers are float32-normal.
    count, mean = 64, 19.8
    x = np.linspace(0.0, 4.5, count)
    y = mean - np.sqrt(mean**2 + (mean - 4.5) ** 2 - (x - mean) ** 2)
    y[[0, -1]] = [4.5, 0.0]
    y_axis = np.sort(y)
    mesh = np.meshgrid(x, y_axis, indexing="ij")
    env = Environment()
    env.place_bin_centers_ = np.column_stack([axis.ravel() for axis in mesh])
    env.centers_shape_ = (count, count)
    env.is_track_interior_ = np.zeros((count, count), dtype=bool)
    env.is_track_interior_[np.arange(count), np.searchsorted(y_axis, y)] = True
    env.is_track_interior_[0, 0] = True
    with pytest.raises(UnsupportedTransitionError, match="float32"):
        build_transition_operator(
            [[RandomWalk(movement_mean=mean, movement_var=1.0)]], [env], (count**2,)
        )


@pytest.mark.unit
def test_original_float64_pdf_overflow_is_rejected_or_preserved_by_dense_fallback():
    env = environment((2, 2), holes=False)
    transition = RandomWalk(movement_var=1e-310)
    with np.errstate(all="ignore"):
        with pytest.raises(UnsupportedTransitionError, match="original Gaussian PDF"):
            build_transition_operator([[transition]], [env], (4,))
        dense = transition.make_state_transition([env])
        operator = build_transition_operator(
            [[transition]], [env], (4,), allow_dense_fallback=True, max_dense_bytes=128
        )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)


def dense_reference(
    transitions, envs, sizes, observations=None, local_std=None, **context
):
    """Call the baseline constructor itself, including scalar block overrides."""
    observations = observations or [ObservationModel() for _ in sizes]
    model = SimpleNamespace(
        state_ind_=np.repeat(np.arange(len(sizes)), sizes),
        environments=tuple(envs),
        observation_models=observations,
        local_position_std=local_std,
    )
    _DetectorBase.initialize_continuous_state_transition(model, transitions, **context)
    return model.continuous_state_transitions_, model.state_ind_


def default_layout(local_std=None):
    env = environment()
    defaults = _ModelDefaults.non_local_defaults()
    observations = defaults["observation_models"]()
    transitions = defaults["continuous_transition_types"]()
    count = len(env.place_bin_centers_)
    sizes = (1 if local_std is None else count, 1, count, count)
    mask = np.concatenate(
        [
            np.ones(1, dtype=bool) if n == 1 else env.is_track_interior_.ravel()
            for n in sizes
        ]
    )
    return env, observations, transitions, sizes, mask


@pytest.mark.unit
@pytest.mark.parametrize("allow_dense_fallback", [False, True])
def test_scalar_to_spatial_transition_without_environment_is_rejected(
    allow_dense_fallback,
):
    """Like the dense constructor, never broadcast a scalar Discrete block."""
    env = environment()
    count = len(env.place_bin_centers_)
    transitions = [[Discrete(), Discrete()], [Discrete(), RandomWalk()]]
    with pytest.raises(ValueError, match="environment_name"):
        dense_reference(transitions, [env], (1, count))
    with pytest.raises(ValueError, match="environment_name"):
        build_transition_operator(
            transitions,
            [env],
            (1, count),
            allow_dense_fallback=allow_dense_fallback,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "transitions",
    [
        [[Uniform("missing")]],
        [[RandomWalk("missing")]],
        [[Discrete(), Uniform("missing")], [Discrete(), Identity("track")]],
    ],
)
def test_unknown_environment_name_lists_available_names(transitions):
    """Dense and structured construction report a missing environment alike."""
    env = environment(name="track")
    n_bins = len(env.place_bin_centers_)
    sizes = (n_bins,) if len(transitions) == 1 else (1, n_bins)
    with pytest.raises(ValidationError, match=r"'missing'[\s\S]*\['track'\]"):
        dense_reference(transitions, [env], sizes)
    with pytest.raises(ValidationError, match=r"'missing'[\s\S]*\['track'\]"):
        build_transition_operator(transitions, [env], sizes)


@pytest.mark.unit
def test_custom_transition_without_context_gets_a_migration_hint():
    """A transition written before ``context`` existed fails with a hint."""

    class OldStyle:
        environment_name = ""

        def make_state_transition(self, environments):
            return np.eye(len(environments[0].place_bin_centers_))

    env = environment((3, 4))
    match = r"OldStyle\.make_state_transition[\s\S]*context=None"
    with pytest.raises(ValidationError, match=match):
        dense_reference([[OldStyle()]], [env], (12,))
    with pytest.raises(ValidationError, match=match):
        build_transition_operator(
            [[OldStyle()]], [env], (12,), allow_dense_fallback=True
        )


@pytest.mark.unit
def test_cross_environment_discrete_with_multibin_local_is_rejected():
    """Dense and structured construction share the multi-bin Local rule."""
    env1, env2 = environment((5,), "one"), environment((5,), "two")
    observations = [
        ObservationModel("one", is_local=True),
        ObservationModel("two"),
    ]
    transitions = [[Discrete(), Discrete()], [Uniform("two", "one"), Identity("two")]]
    match = (
        r"^Cross-environment Discrete\(\) transition between 'one' and 'two' "
        r"is not supported with multi-bin local \(local_position_std is set\)\. "
        r"Use explicit Uniform\(environment_name=\.\.\., "
        r"environment2_name=\.\.\.\) instead\.$"
    )
    with pytest.raises(ValueError, match=match):
        dense_reference(transitions, [env1, env2], (5, 5), observations, 2.0)
    with pytest.raises(ValueError, match=match):
        build_transition_operator(transitions, [env1, env2], (5, 5), observations, 2.0)


def assert_products(operator, dense, dtype):
    rng = np.random.default_rng(20261006)
    left = rng.uniform(0.01, 1.0, dense.shape[0]).astype(dtype)
    right = rng.normal(size=dense.shape[1]).astype(dtype)
    dense = jnp.asarray(dense, dtype=dtype)
    left, right = jnp.asarray(left), jnp.asarray(right)
    tolerance = (
        {"rtol": 1e-6, "atol": 1e-7}
        if dtype == np.float32
        else {"rtol": 1e-12, "atol": 1e-14}
    )
    np.testing.assert_allclose(
        jax.jit(operator.forward)(left), left @ dense, **tolerance
    )
    np.testing.assert_allclose(
        jax.jit(operator.backward)(right), dense @ right, **tolerance
    )
    assert operator.forward(left).dtype == left.dtype


def product_operator(operator, fused):
    """The block operator, or its fused product-only form."""
    return operator.fused() if fused else operator


@pytest.mark.unit
@pytest.mark.parametrize("kind", [Uniform, Identity, Discrete])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_simple_exact_products(kind, dtype):
    with precision_mode(dtype == np.float64):
        env = environment((1,) if kind is Discrete else (11,))
        sizes = (len(env.place_bin_centers_),)
        transitions = [[kind()]]
        dense, _ = dense_reference(transitions, [env], sizes)
        operator = build_transition_operator(transitions, [env], sizes)
        assert_products(operator, dense, dtype)
        assert_products(operator.fused(), dense, dtype)
        np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(7,), (5, 7), (4, 3, 2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_separable_gaussian_products_with_holes_nonzero_mean_and_diagonal_covariance(
    shape, dtype
):
    with precision_mode(dtype == np.float64):
        env = environment(shape)
        variance = np.linspace(0.8, 2.5, len(shape))
        transitions = [[RandomWalk(movement_var=np.diag(variance), movement_mean=0.25)]]
        sizes = (len(env.place_bin_centers_),)
        dense, _ = dense_reference(transitions, [env], sizes)
        operator = build_transition_operator(transitions, [env], sizes)
        assert_products(operator, dense, dtype)
        assert_products(operator.fused(), dense, dtype)
        np.testing.assert_allclose(
            np.asarray(LazyDenseTransition(operator)), dense, rtol=1e-12, atol=1e-14
        )


@pytest.mark.unit
@pytest.mark.parametrize("kind", [Uniform, Identity, RandomWalk])
def test_empty_destination_and_gaussian_underflow_preserve_zero_rows(kind):
    env = environment((4, 5))
    env.is_track_interior_[:] = False
    transitions = [[kind()]]
    dense, _ = dense_reference(transitions, [env], (20,))
    operator = build_transition_operator(transitions, [env], (20,))
    for product in (operator, operator.fused()):
        np.testing.assert_array_equal(product.forward(jnp.ones(20)), np.zeros(20))
        np.testing.assert_array_equal(product.backward(jnp.ones(20)), np.zeros(20))
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)
    if kind is RandomWalk:
        env.is_track_interior_[:] = True
        transitions = [[RandomWalk(movement_mean=1e3, movement_var=1.0)]]
        operator = build_transition_operator(transitions, [env], (20,))
        dense, _ = dense_reference(transitions, [env], (20,))
        np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)


@pytest.mark.integration
@pytest.mark.parametrize("local_std", [None, 2.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_default_four_state_and_multibin_local_match_original_masks_and_rectangular_blocks(
    local_std, dtype
):
    with precision_mode(dtype == np.float64):
        env, observations, transitions, sizes, mask = default_layout(local_std)
        dense, state_ind = dense_reference(
            transitions, [env], sizes, observations, local_std
        )
        operator = build_transition_operator(
            transitions, [env], sizes, observations, local_std
        )
        weights = np.array(
            [
                [0.6, 0, 0.1, 0.3],
                [0.1, 0.7, 0.1, 0.1],
                [0.2, 0.1, 0.65, 0.05],
                [0, 0.1, 0.3, 0.6],
            ]
        )
        joint = dense * weights[np.ix_(state_ind, state_ind)]
        for fused in (False, True):
            assert_products(
                product_operator(operator.bind_discrete(weights), fused), joint, dtype
            )
            assert_products(
                product_operator(
                    operator.restricted(mask).bind_discrete(weights), fused
                ),
                joint[np.ix_(mask, mask)],
                dtype,
            )
        np.testing.assert_allclose(
            joint[np.ix_(mask, mask)].sum(axis=1), 1.0, atol=1e-14
        )


def pair_reference(continuous, state_ind, weights, filtered, ratio):
    """``w[i, j] * sum over x in i, y in j of filtered[x] C[x, y] ratio[y]``."""
    n_states = weights.shape[0]
    xi = filtered[:, None] * continuous * ratio[None, :]
    onehot = np.eye(n_states)[state_ind]
    return weights * (onehot.T @ xi @ onehot)


@pytest.mark.integration
@pytest.mark.parametrize("local_std", [None, 2.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fused", [False, True])
def test_pair_products_aggregate_each_state_pair(local_std, dtype, fused):
    """Per-step discrete transition statistics from the structured blocks."""
    with precision_mode(dtype == np.float64):
        env, observations, transitions, sizes, mask = default_layout(local_std)
        dense, state_ind = dense_reference(
            transitions, [env], sizes, observations, local_std
        )
        operator = build_transition_operator(
            transitions, [env], sizes, observations, local_std
        )
        weights = np.array(
            [
                [0.6, 0, 0.1, 0.3],
                [0.1, 0.7, 0.1, 0.1],
                [0.2, 0.1, 0.65, 0.05],
                [0, 0.1, 0.3, 0.6],
            ]
        )
        rng = np.random.default_rng(7)
        tolerance = (
            {"rtol": 1e-5, "atol": 1e-7}
            if dtype == np.float32
            else {"rtol": 1e-12, "atol": 1e-14}
        )
        for restriction in (None, mask):
            selected = (
                np.arange(len(state_ind))
                if restriction is None
                else np.flatnonzero(restriction)
            )
            base = operator if restriction is None else operator.restricted(mask)
            filtered = rng.uniform(0.0, 1.0, len(selected))
            ratio = rng.uniform(0.0, 2.0, len(selected))
            expected = pair_reference(
                dense[np.ix_(selected, selected)],
                state_ind[selected],
                weights,
                filtered,
                ratio,
            )
            args = (
                jnp.asarray(filtered, dtype=dtype),
                jnp.asarray(ratio, dtype=dtype),
            )
            bound = product_operator(base.bind_discrete(weights), fused)
            np.testing.assert_allclose(
                jax.jit(bound.pair_products)(*args), expected, **tolerance
            )
            per_step = product_operator(base, fused)
            actual = jax.jit(per_step.pair_products)(
                *args, jnp.asarray(weights, dtype=dtype)
            )
            assert actual.dtype == dtype
            np.testing.assert_allclose(actual, expected, **tolerance)


@pytest.mark.unit
def test_rectangular_multienvironment_uniform_blocks():
    env1, env2 = environment((4, 3), "one"), environment((3, 5), "two")
    transitions = [
        [Identity("one"), Uniform("one", "two")],
        [Uniform("two", "one"), Identity("two")],
    ]
    sizes = (12, 15)
    observations = [ObservationModel("one"), ObservationModel("two")]
    dense, _ = dense_reference(transitions, [env1, env2], sizes, observations)
    operator = build_transition_operator(
        transitions, [env1, env2], sizes, observation_models=observations
    )
    assert_products(operator, dense, np.float32)


@pytest.mark.unit
def test_scalar_source_uses_the_destination_states_environment():
    """From a one-bin state in "one", Uniform("one", "two") into a state in
    "two" is uniform over "two"'s interior (not "one"'s), dense and
    structured."""
    env1, env2 = environment((5,), "one"), environment((5,), "two")
    env1.is_track_interior_[:] = [True, False, True, True, False]
    env2.is_track_interior_[:] = [False, True, False, False, True]
    observations = [ObservationModel("one", is_local=True), ObservationModel("two")]
    transitions = [[Discrete(), Uniform("one", "two")], [Discrete(), Identity("two")]]
    dense, _ = dense_reference(transitions, [env1, env2], (1, 5), observations)
    np.testing.assert_array_equal(dense[0, 1:], [0.0, 0.5, 0.0, 0.0, 0.5])
    operator = build_transition_operator(
        transitions, [env1, env2], (1, 5), observation_models=observations
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)


@pytest.mark.unit
@pytest.mark.parametrize(
    "transition",
    [
        RandomWalk(use_manifold_distance=True),
        RandomWalk(movement_var=np.array([[1.0, 0.2], [0.2, 1.0]])),
        RandomWalkDirection1(),
        EmpiricalMovement(),
    ],
)
def test_unsupported_capabilities_fail_explicitly(transition):
    with pytest.raises(UnsupportedTransitionError):
        build_transition_operator([[transition]], [environment((4, 5))], (20,))


@pytest.mark.unit
def test_dense_fallback_budget_is_cumulative_and_checked_before_any_dense_call(
    monkeypatch,
):
    env = environment((4, 5))
    transition = RandomWalk(movement_var=np.array([[1.0, 0.2], [0.2, 1.0]]))

    def forbidden(*args, **kwargs):
        pytest.fail("Budget overflow must fail before constructing ANY dense block")

    monkeypatch.setattr(RandomWalk, "make_state_transition", forbidden)
    with pytest.raises(DenseTransitionBudgetError):
        build_transition_operator(
            [[transition] * 2] * 2,
            [env],
            (20, 20),
            allow_dense_fallback=True,
            max_dense_bytes=3 * 20 * 20 * 8,
        )


@pytest.mark.unit
def test_explicit_full_covariance_fallback_preserves_model_and_reports_storage():
    env = environment((4, 5))
    transition = RandomWalk(movement_var=np.array([[1.0, 0.2], [0.2, 1.0]]))
    dense, _ = dense_reference([[transition]], [env], (20,))
    operator = build_transition_operator(
        [[transition]],
        [env],
        (20,),
        allow_dense_fallback=True,
        max_dense_bytes=dense.nbytes,
    )
    assert_products(operator, dense, np.float32)
    assert operator.dense_fallback_nbytes == dense.nbytes


@pytest.mark.unit
def test_supported_construction_and_pickle_never_call_dense_builders(monkeypatch):
    env = environment((182, 182), holes=False)
    count = len(env.place_bin_centers_)
    defaults = _ModelDefaults.non_local_defaults()

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Supported structured construction must not call dense constructors"
        )

    for cls in (RandomWalk, Uniform, Identity, Discrete):
        monkeypatch.setattr(cls, "make_state_transition", forbidden)
    operator = build_transition_operator(
        defaults["continuous_transition_types"](),
        [env],
        (1, 1, count, count),
        defaults["observation_models"](),
    )
    assert operator.storage_nbytes < 8 * 1024**2
    view = LazyDenseTransition(operator, max_bytes=1024)
    payload = pickle.dumps(view)
    assert len(payload) < 8 * 1024**2
    restored = pickle.loads(payload)
    assert restored.shape == view.shape
    with pytest.raises(DenseTransitionBudgetError):
        np.asarray(restored)
    # A small indexed read must not require the entire combined matrix.
    np.testing.assert_array_equal(view[:2, :2], np.ones((2, 2)))
    # A full-shape Boolean mask selects elements, as in NumPy.
    small = LazyDenseTransition(operator.restricted(np.arange(operator.n_bins) < 3))
    mask = np.eye(3, dtype=bool) | np.eye(3, k=1, dtype=bool)
    np.testing.assert_array_equal(small[mask], np.asarray(small)[mask])


@pytest.mark.unit
@pytest.mark.parametrize("fused", [False, True])
def test_runtime_products_and_discrete_weights_have_correct_gradients(fused):
    with precision_mode(True):
        env, observations, transitions, sizes, mask = default_layout()
        dense, state_ind = dense_reference(transitions, [env], sizes, observations)
        operator = product_operator(
            build_transition_operator(
                transitions, [env], sizes, observations
            ).restricted(mask),
            fused,
        )
        dense = jnp.asarray(dense[np.ix_(mask, mask)])
        state_ind = state_ind[mask]
        rng = np.random.default_rng(10)
        x = jnp.asarray(rng.uniform(0.1, 1, mask.sum()))
        y = jnp.asarray(rng.normal(size=mask.sum()))
        weights = jnp.asarray(rng.uniform(0.1, 1, (4, 4)))

        def objective(x, y, w):
            return jnp.sum(operator.forward(x, w) * y) + jnp.sum(
                operator.backward(y, w) ** 2
            )

        def reference(x, y, w):
            matrix = dense * w[np.ix_(state_ind, state_ind)]
            return jnp.sum((x @ matrix) * y) + jnp.sum((matrix @ y) ** 2)

        actual = jax.grad(objective, argnums=(0, 1, 2))(x, y, weights)
        expected = jax.grad(reference, argnums=(0, 1, 2))(x, y, weights)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-13)


@pytest.mark.unit
def test_float32_gaussian_preserves_float64_normalized_rows_when_raw_pdf_is_tiny():
    env = environment((20,), holes=False)
    transitions = [[RandomWalk(movement_mean=30.0, movement_var=0.8)]]
    dense, _ = dense_reference(transitions, [env], (20,))
    operator = build_transition_operator(transitions, [env], (20,))
    assert_products(operator, dense, np.float32)


@pytest.mark.unit
def test_subnormal_gaussian_normalization_is_explicitly_unsupported_or_budgeted():
    env = environment((2,), holes=False)
    transition = RandomWalk(movement_mean=39.0, movement_var=1.0)
    with pytest.raises(UnsupportedTransitionError, match="Subnormal"):
        build_transition_operator([[transition]], [env], (2,))
    operator = build_transition_operator(
        [[transition]], [env], (2,), allow_dense_fallback=True, max_dense_bytes=32
    )
    dense, _ = dense_reference([[transition]], [env], (2,))
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)


@pytest.mark.unit
def test_fallback_receives_original_empirical_context_and_defaults():
    env = environment((5,), holes=False)
    env.edges_ = [np.arange(6.0) - 0.5]
    env.position_range = ((-0.5, 4.5),)
    position = np.resize(np.arange(5.0), 30)[:, None]
    context = {
        "position": position,
        "is_training": np.arange(30) % 3 != 0,
        "encoding_group_labels": np.zeros(30),
        "environment_labels": np.full(30, ""),
    }
    transitions = [[EmpiricalMovement()]]
    dense, _ = dense_reference(transitions, [env], (5,), **context)
    operator = build_transition_operator(
        transitions,
        [env],
        (5,),
        allow_dense_fallback=True,
        context=TransitionContext(**context),
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)
    # The context defaults absent labels to group zero, as the detector fit
    # always has.
    transitions = [[EmpiricalMovement(encoding_group=1)]]
    dense, _ = dense_reference(transitions, [env], (5,), position=position)
    operator = build_transition_operator(
        transitions,
        [env],
        (5,),
        allow_dense_fallback=True,
        context=TransitionContext(position),
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)


@pytest.mark.unit
@pytest.mark.parametrize("n_bins", [5, 8], ids=["no_holes", "holes"])
def test_empirical_movement_scalar_blocks_match_dense_constructor(n_bins):
    """Scalar-to-spatial and spatial-to-scalar EmpiricalMovement blocks agree."""
    env = environment((n_bins,))
    interior = env.is_track_interior_.ravel()
    assert interior.all() == (n_bins == 5)
    env.edges_ = (np.arange(n_bins + 1.0) - 0.5,)
    position = np.resize(np.arange(float(n_bins)), 6 * n_bins)[:, None]
    transitions = [
        [Discrete(), EmpiricalMovement()],
        [EmpiricalMovement(), EmpiricalMovement()],
    ]
    sizes = (1, n_bins)
    dense, _ = dense_reference(transitions, [env], sizes, position=position)
    operator = build_transition_operator(
        transitions,
        [env],
        sizes,
        allow_dense_fallback=True,
        context=TransitionContext(position),
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)
    np.testing.assert_array_equal(dense[0, 1:], interior / interior.sum())
    np.testing.assert_array_equal(dense[1:, 0], np.ones(n_bins))


@pytest.mark.unit
@pytest.mark.parametrize("kind", [Uniform, RandomWalk])
def test_large_finite_backward_values_do_not_overflow_before_normalization(kind):
    env = environment((5, 7), holes=False)
    transitions = [[kind()]]
    dense, _ = dense_reference(transitions, [env], (35,))
    operator = build_transition_operator(transitions, [env], (35,))
    values = jnp.full(35, 1e38, dtype=jnp.float32)
    np.testing.assert_allclose(
        operator.backward(values),
        jnp.asarray(dense, dtype=jnp.float32) @ values,
        rtol=1e-6,
        atol=0,
    )


def structured_hmm(operator, initial, likelihood, weights):
    from non_local_detector.core import _condition_on, _divide_safe, _normalize

    def forward(carry, observation):
        evidence, prediction = carry
        ll, discrete = observation
        filtered, increment = _condition_on(prediction, ll)
        return (evidence + increment, operator.forward(filtered, discrete)), (
            filtered,
            prediction,
        )

    final, (filtered, predictions) = jax.lax.scan(
        forward, (0.0, initial), (likelihood, weights)
    )
    n_time = len(likelihood)

    def backward(smoothed_next, observation):
        filtered_t, discrete, time = observation
        predicted_next = operator.forward(filtered_t, discrete)
        smoothed = filtered_t * operator.backward(
            _divide_safe(smoothed_next, predicted_next), discrete
        )
        smoothed = jnp.where(time == n_time - 1, filtered_t, smoothed)
        smoothed, _ = _normalize(smoothed, axis=-1)
        return smoothed, smoothed

    _, smoothed = jax.lax.scan(
        backward, filtered[-1], (filtered, weights, jnp.arange(n_time)), reverse=True
    )
    return final, filtered, predictions, smoothed


@pytest.mark.integration
@pytest.mark.parametrize("covariate", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "case", ["ordinary", "singleton", "missing", "impossible", "nan", "zero_support"]
)
@pytest.mark.parametrize("fused", [False, True])
def test_hmm_filter_smoother_and_evidence_match_dense_reference(
    covariate, dtype, case, fused
):
    from non_local_detector.core import (
        _filter_covariate_dependent_impl,
        _filter_impl,
        _smoother_covariate_dependent_impl,
        _smoother_impl,
    )

    with precision_mode(dtype == np.float64):
        env, observations, transitions, sizes, mask = default_layout()
        dense, state_ind = dense_reference(transitions, [env], sizes, observations)
        dense = jnp.asarray(dense[np.ix_(mask, mask)], dtype=dtype)
        state_ind = jnp.asarray(state_ind[mask])
        operator = product_operator(
            build_transition_operator(
                transitions, [env], sizes, observations
            ).restricted(mask),
            fused,
        )
        rng = np.random.default_rng(100)
        n_time = 1 if case == "singleton" else 17
        likelihood = rng.normal(size=(n_time, mask.sum())).astype(dtype)
        discrete = rng.uniform(0.1, 1, (n_time if covariate else 1, 4, 4)).astype(dtype)
        discrete /= discrete.sum(axis=2, keepdims=True)
        initial = rng.uniform(0.01, 1, mask.sum()).astype(dtype)
        initial /= initial.sum()
        if case == "missing":
            likelihood[4:8] = 0
        elif case == "impossible":
            likelihood[6] = -np.inf
        elif case == "nan":
            likelihood[8, 3] = np.nan
        elif case == "zero_support":
            initial[:] = 0
            initial[0] = 1
            discrete[:] = np.eye(4)
            likelihood[0, 0] = -np.inf
        initial, likelihood, discrete = (
            jnp.asarray(initial),
            jnp.asarray(likelihood),
            jnp.asarray(discrete),
        )
        if covariate:
            expected_final, (filtered, predicted, _) = _filter_covariate_dependent_impl(
                initial, discrete, dense, state_ind, likelihood
            )
            smoothed = _smoother_covariate_dependent_impl(
                discrete, dense, state_ind, filtered
            )
            weights = discrete
        else:
            joint = (
                dense
                * discrete[0][np.ix_(np.asarray(state_ind), np.asarray(state_ind))]
            )
            expected_final, (filtered, predicted, _) = _filter_impl(
                initial, joint, likelihood
            )
            smoothed = _smoother_impl(joint, filtered)
            weights = jnp.broadcast_to(discrete[0], (n_time, 4, 4))
        actual = jax.jit(structured_hmm)(operator, initial, likelihood, weights)
        expected = (expected_final, filtered, predicted, smoothed)
        tolerance = (
            {"rtol": 1e-6, "atol": 1e-6}
            if dtype == np.float32
            else {"rtol": 1e-12, "atol": 1e-13}
        )
        for left, right in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(left, right, equal_nan=True, **tolerance)


@pytest.mark.integration
@pytest.mark.parametrize("fused", [False, True])
def test_hmm_gradients_through_nonuniform_prior_likelihood_and_discrete_parameters(
    fused,
):
    from non_local_detector.core import _filter_covariate_dependent_impl

    with precision_mode(True):
        env, observations, transitions, sizes, mask = default_layout()
        dense, state_ind = dense_reference(transitions, [env], sizes, observations)
        dense = jnp.asarray(dense[np.ix_(mask, mask)])
        state_ind = jnp.asarray(state_ind[mask])
        operator = product_operator(
            build_transition_operator(
                transitions, [env], sizes, observations
            ).restricted(mask),
            fused,
        )
        rng = np.random.default_rng(8)
        logits = jnp.asarray(rng.normal(size=mask.sum()))
        likelihood = jnp.asarray(rng.normal(size=(5, mask.sum())))
        weights = jnp.asarray(rng.normal(size=(5, 4, 4)))

        def objective(prior, ll, parameters):
            probabilities = jax.nn.softmax(prior)
            transitions = jax.nn.softmax(parameters, axis=-1)
            return structured_hmm(operator, probabilities, ll, transitions)[0][0]

        def reference(prior, ll, parameters):
            return _filter_covariate_dependent_impl(
                jax.nn.softmax(prior),
                jax.nn.softmax(parameters, axis=-1),
                dense,
                state_ind,
                ll,
            )[0][0]

        actual = jax.grad(objective, argnums=(0, 1, 2))(logits, likelihood, weights)
        expected = jax.grad(reference, argnums=(0, 1, 2))(logits, likelihood, weights)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-13)


@pytest.mark.unit
@pytest.mark.parametrize("all_impossible", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_hmm_boundary_gradients_match_dense_reference(all_impossible, fused):
    from non_local_detector.core import _filter_covariate_dependent_impl

    with precision_mode(True):
        env, observations, transitions, sizes, mask = default_layout()
        dense, state_ind = dense_reference(transitions, [env], sizes, observations)
        dense = jnp.asarray(dense[np.ix_(mask, mask)])
        state_ind = jnp.asarray(state_ind[mask])
        operator = product_operator(
            build_transition_operator(
                transitions, [env], sizes, observations
            ).restricted(mask),
            fused,
        )
        prior = jnp.zeros(mask.sum()).at[0].set(1.0)
        ll = (
            jnp.full((3, mask.sum()), -jnp.inf)
            if all_impossible
            else jnp.zeros((3, mask.sum())).at[0, 0].set(-jnp.inf)
        )
        weights = jnp.broadcast_to(jnp.eye(4), (3, 4, 4))

        def objective(initial, likelihood, discrete):
            return structured_hmm(operator, initial, likelihood, discrete)[0][0]

        def reference(initial, likelihood, discrete):
            return _filter_covariate_dependent_impl(
                initial, discrete, dense, state_ind, likelihood
            )[0][0]

        actual = jax.grad(objective, argnums=(0, 1, 2))(prior, ll, weights)
        expected = jax.grad(reference, argnums=(0, 1, 2))(prior, ll, weights)
        for left, right in zip(actual, expected, strict=True):
            assert np.all(np.isfinite(left))
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-13)


@pytest.mark.unit
@pytest.mark.parametrize("dense_gaussian_bytes", [None, 0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_gaussian_blocks_within_budget_are_dense(
    dense_gaussian_bytes, dtype, monkeypatch
):
    """A small restricted Gaussian block is stored dense (one matrix-vector
    product per step instead of several small kernels); above the budget it
    keeps its separable product. Both match the dense reference."""
    from non_local_detector import transition_operators

    if dense_gaussian_bytes is not None:
        monkeypatch.setattr(
            transition_operators, "_DENSE_GAUSSIAN_BLOCK_BYTES", dense_gaussian_bytes
        )
    with precision_mode(dtype == np.float64):
        env = environment((6, 7))
        transitions = [[RandomWalk(movement_var=np.diag([0.9, 1.7]))]]
        sizes = (len(env.place_bin_centers_),)
        dense, _ = dense_reference(transitions, [env], sizes)
        mask = env.is_track_interior_.ravel()
        operator = build_transition_operator(transitions, [env], sizes).restricted(mask)
        fused = operator.fused()
        expected = (
            transition_operators.DenseBlock
            if dense_gaussian_bytes is None
            else transition_operators.RestrictedBlock
        )
        assert [type(block) for block in fused.blocks] == [expected]
        assert_products(fused, dense[np.ix_(mask, mask)], dtype)
        rng = np.random.default_rng(7)
        filtered = jnp.asarray(rng.uniform(0.01, 1, mask.sum()), dtype=dtype)
        ratio = jnp.asarray(rng.uniform(0.5, 2, mask.sum()), dtype=dtype)
        np.testing.assert_allclose(
            fused.pair_products(filtered, ratio),
            operator.pair_products(filtered, ratio),
            rtol=1e-6 if dtype == np.float32 else 1e-12,
        )


@pytest.mark.unit
@pytest.mark.parametrize("n_dense", [0, 2, 4])
def test_dense_gaussian_blocks_count_against_the_dense_budget(n_dense):
    """Gaussian blocks stored dense are retained dense matrices, so together
    they stay within ``max_dense_bytes``; the rest keep separable products."""
    from non_local_detector import transition_operators

    env = environment((6, 7))
    n = len(env.place_bin_centers_)
    walk = RandomWalk(movement_var=np.diag([0.9, 1.7]))
    transitions = [[walk, walk], [walk, walk]]
    dense, _ = dense_reference(transitions, [env], (n, n))
    mask = np.tile(env.is_track_interior_.ravel(), 2)
    operator = build_transition_operator(transitions, [env], (n, n)).restricted(mask)
    block_bytes = int(mask[:n].sum()) ** 2 * 8
    fused = operator.fused(max_dense_bytes=n_dense * block_bytes)
    kinds = [type(block) for block in fused.blocks]
    assert kinds.count(transition_operators.DenseBlock) == n_dense
    assert kinds.count(transition_operators.RestrictedBlock) == 4 - n_dense
    assert_products(fused, dense[np.ix_(mask, mask)], np.float32)
    rng = np.random.default_rng(8)
    filtered = jnp.asarray(rng.uniform(0.01, 1, mask.sum()), dtype=np.float32)
    ratio = jnp.asarray(rng.uniform(0.5, 2, mask.sum()), dtype=np.float32)
    np.testing.assert_allclose(
        fused.pair_products(filtered, ratio),
        operator.pair_products(filtered, ratio),
        rtol=1e-6,
    )


@pytest.mark.unit
@pytest.mark.parametrize("max_dense_bytes", [0, DEFAULT_MAX_DENSE_BYTES])
def test_backward_and_pairs_matches_separate_products(max_dense_bytes):
    """The smoother's combined product reuses each block's backward product
    for the transition statistics, with separable or dense Gaussian blocks."""
    env = environment((6, 7))
    n = len(env.place_bin_centers_)
    walk = RandomWalk(movement_var=np.diag([0.9, 1.7]))
    transitions = [[walk, Uniform()], [Uniform(), walk]]
    mask = np.tile(env.is_track_interior_.ravel(), 2)
    fused = (
        build_transition_operator(transitions, [env], (n, n))
        .restricted(mask)
        .bind_discrete(np.array([[0.9, 0.1], [0.2, 0.8]]))
        .fused(max_dense_bytes)
    )
    rng = np.random.default_rng(9)
    filtered = jnp.asarray(rng.uniform(0.01, 1, mask.sum()), dtype=np.float32)
    ratio = jnp.asarray(rng.uniform(0.5, 2, mask.sum()), dtype=np.float32)
    backward, pairs = jax.jit(fused.backward_and_pairs)(ratio, filtered)
    np.testing.assert_allclose(backward, fused.backward(ratio), rtol=1e-6)
    np.testing.assert_allclose(pairs, fused.pair_products(filtered, ratio), rtol=1e-6)


@pytest.mark.unit
def test_unsupported_transition_error_names_the_representation_setting():
    """A transition without a structured operator points at the user setting
    that permits its dense block."""
    with pytest.raises(
        UnsupportedTransitionError, match='transition_representation="auto"'
    ):
        build_transition_operator([[EmpiricalMovement()]], [environment((4, 5))], (20,))


@pytest.mark.unit
def test_dense_budget_error_names_the_budget_setting():
    """Exceeding the dense fallback budget names max_dense_transition_bytes."""
    transition = RandomWalk(movement_var=np.array([[1.0, 0.2], [0.2, 1.0]]))
    with pytest.raises(DenseTransitionBudgetError, match="max_dense_transition_bytes"):
        build_transition_operator(
            [[transition]],
            [environment((4, 5))],
            (20,),
            allow_dense_fallback=True,
            max_dense_bytes=8,
        )


@pytest.mark.unit
def test_opaque_custom_error_names_the_dense_representation():
    """An opaque custom transition on deferred distances points at
    transition_representation="dense"."""

    class Custom:
        environment_name = ""

        def make_state_transition(self, environments, context=None):
            raise AssertionError

    position = np.array([[0.0, 0.0], [3.0, 4.0]])
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False, compute_all_pairs_distances=False
    )
    with pytest.raises(
        UnsupportedTransitionError, match='transition_representation="dense"'
    ):
        build_transition_operator(
            [[Custom()]],
            [env],
            (len(env.place_bin_centers_),),
            allow_dense_fallback=True,
        )


def _custom_returning(matrix):
    """A custom transition whose dense block is ``matrix``."""

    class Custom:
        environment_name = ""

        def make_state_transition(self, environments, context=None):
            return matrix

    return Custom()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        (np.ones((1, 1)), "shape"),
        (np.ones((12,)), "shape"),
        (np.full((12, 12), 2.0 / 12), "at most 1"),
        (np.eye(12) * 2.0 - np.roll(np.eye(12), 1, axis=1), "non-negative"),
        (np.full((12, 12), np.nan), "non-finite"),
    ],
    ids=["broadcast-scalar", "one-dimensional", "rows-sum-to-two", "negative", "nan"],
)
def test_invalid_custom_block_is_rejected_in_dense_and_operator_paths(matrix, match):
    """A custom transition's block must be its states' shape, non-negative,
    and have rows summing to at most 1; otherwise both constructions fail
    instead of broadcasting it into the transition."""
    env = environment((3, 4))
    with pytest.raises(ValueError, match=match):
        dense_reference([[_custom_returning(matrix)]], [env], (12,))
    with pytest.raises(ValueError, match=match):
        build_transition_operator(
            [[_custom_returning(matrix)]], [env], (12,), allow_dense_fallback=True
        )


@pytest.mark.unit
def test_custom_block_with_empty_rows_is_accepted():
    """Rows summing to 0 (bins nothing can leave) are valid."""
    matrix = np.eye(12)
    matrix[[0, 3]] = 0.0
    env = environment((3, 4))
    dense, _ = dense_reference([[_custom_returning(matrix)]], [env], (12,))
    np.testing.assert_array_equal(dense, matrix)
    operator = build_transition_operator(
        [[_custom_returning(matrix)]], [env], (12,), allow_dense_fallback=True
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), matrix)


@pytest.mark.unit
def test_spatial_discrete_block_is_rejected_instead_of_broadcast():
    """A one-bin Discrete block between two spatial states is not broadcast
    into a matrix of ones."""
    env = environment((3, 4))
    match = r"Discrete gives a block with the wrong shape"
    with pytest.raises(ValidationError, match=match):
        dense_reference([[Discrete()]], [env], (12,))
    for allow_dense_fallback in (False, True):
        with pytest.raises(ValidationError, match=match):
            build_transition_operator(
                [[Discrete()]], [env], (12,), allow_dense_fallback=allow_dense_fallback
            )


@pytest.mark.unit
def test_transition_without_environment_name_is_rejected():
    """A continuous transition must declare environment_name."""

    class Nameless:
        def make_state_transition(self, environments, context=None):
            return np.eye(12)

    env = environment((3, 4))
    with pytest.raises(ValidationError, match="environment_name"):
        dense_reference([[Nameless()]], [env], (12,))
    with pytest.raises(ValidationError, match="environment_name"):
        build_transition_operator(
            [[Nameless()]], [env], (12,), allow_dense_fallback=True
        )


@pytest.mark.unit
def test_custom_transition_opts_into_a_structured_block():
    """A custom class implementing SupportsStructuredBlock gets its block
    without a dense constructor, even with no dense fallback allowed."""
    from non_local_detector.transition_operators import (
        IdentityBlock,
        SupportsStructuredBlock,
    )

    class Stay:
        environment_name = ""

        def make_state_transition(self, environments, context=None):
            msg = "the structured block must be used"
            raise AssertionError(msg)

        def structured_block(self, environments):
            return IdentityBlock(environments[0].is_track_interior_flat_)

        def dense_block_shape(self, environments):
            n_bins = len(environments[0].place_bin_centers_)
            return (n_bins, n_bins)

        def graph_distance_environment(self, environments):
            return None

    env = environment((3, 4))
    assert isinstance(Stay(), SupportsStructuredBlock)
    operator = build_transition_operator([[Stay()]], [env], (12,))
    np.testing.assert_array_equal(
        np.asarray(LazyDenseTransition(operator)),
        np.diag(env.is_track_interior_flat_.astype(float)),
    )


@pytest.mark.unit
def test_subclass_overriding_make_state_transition_falls_back_to_dense():
    """A RandomWalk subclass with its own make_state_transition does not
    inherit the Gaussian structured block."""

    class Stay(RandomWalk):
        def make_state_transition(self, environments, context=None):
            return np.eye(len(environments[0].place_bin_centers_))

    env = environment((3, 4), holes=False)
    with pytest.raises(UnsupportedTransitionError):
        build_transition_operator([[Stay()]], [env], (12,))
    operator = build_transition_operator(
        [[Stay()]], [env], (12,), allow_dense_fallback=True
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), np.eye(12))


@pytest.mark.unit
def test_empirical_movement_with_speedup_leaking_mass_is_accepted():
    """``EmpiricalMovement(speedup=2)`` squares a matrix with an empty row
    (the last sample's bin, never left), so a row that reaches that bin sums
    to less than 1. Both constructors keep such a sub-stochastic block."""
    position = np.concatenate([np.tile(np.arange(2.5, 50, 5.0), 4), [92.5]])[
        :, np.newaxis
    ]
    env = Environment(place_bin_size=5.0, position_range=[(0.0, 100.0)])
    env = env.fit_place_grid(position)
    transition = EmpiricalMovement(speedup=2)
    expected = transition.make_state_transition(
        [env], TransitionContext(position=position)
    )
    row_sums = expected.sum(axis=1)
    assert np.any((row_sums > 0.0) & (row_sums < 1.0 - 1e-2))
    count = len(env.place_bin_centers_)

    dense, _ = dense_reference([[transition]], [env], (count,), position=position)
    np.testing.assert_array_equal(dense, expected)
    operator = build_transition_operator(
        [[transition]],
        [env],
        (count,),
        allow_dense_fallback=True,
        context=TransitionContext(position=position),
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), expected)


@pytest.mark.unit
def test_block_with_rows_above_one_is_rejected():
    """Row sums above 1 beyond the probability tolerance are rejected; the
    hint does not tell the user to renormalize a package transition."""
    from non_local_detector._validation import PROBABILITY_SUM_TOLERANCE

    env = environment((3, 4))
    within = np.eye(12) * (1.0 + PROBABILITY_SUM_TOLERANCE / 2)
    dense, _ = dense_reference([[_custom_returning(within)]], [env], (12,))
    np.testing.assert_array_equal(dense, within)
    above = np.eye(12) * (1.0 + 2 * PROBABILITY_SUM_TOLERANCE)
    with pytest.raises(ValidationError, match="at most 1") as error:
        dense_reference([[_custom_returning(above)]], [env], (12,))
    assert "Normalize" not in str(error.value)


@pytest.mark.unit
def test_subclass_overriding_a_helper_falls_back_to_dense():
    """A RandomWalk subclass that changes a helper of make_state_transition
    is planned as a custom transition, not given RandomWalk's Gaussian
    block."""
    from dataclasses import dataclass

    from non_local_detector.continuous_state_transitions import (
        _euclidean_random_walk,
    )

    @dataclass
    class WideRandomWalk(RandomWalk):
        def _handle_no_track_graph(self, environment):
            return _euclidean_random_walk(
                environment, self.movement_mean, 10 * self.movement_var
            )

    env = environment((3, 4), holes=False)
    transition = WideRandomWalk()
    with pytest.raises(UnsupportedTransitionError):
        build_transition_operator([[transition]], [env], (12,))
    operator = build_transition_operator(
        [[transition]], [env], (12,), allow_dense_fallback=True
    )
    np.testing.assert_array_equal(
        np.asarray(LazyDenseTransition(operator)),
        transition.make_state_transition([env]),
    )


@pytest.mark.unit
def test_subclass_defining_structured_block_uses_it():
    """A subclass opts in again by defining structured_block itself."""
    from dataclasses import dataclass

    from non_local_detector.transition_operators import IdentityBlock

    @dataclass
    class StayingRandomWalk(RandomWalk):
        def make_state_transition(self, environments, context=None):
            return np.diag(environments[0].is_track_interior_flat_.astype(float))

        def structured_block(self, environments):
            return IdentityBlock(environments[0].is_track_interior_flat_)

    env = environment((3, 4))
    operator = build_transition_operator([[StayingRandomWalk()]], [env], (12,))
    assert isinstance(operator.blocks[0][0], IdentityBlock)


@pytest.mark.unit
def test_instance_assigned_make_state_transition_is_custom():
    """A make_state_transition assigned on the instance replaces the class's
    matrix, so the class's structured block no longer describes it."""
    env = environment((3, 4), holes=False)
    transition = RandomWalk()
    transition.make_state_transition = lambda environments, context=None: np.eye(12)
    with pytest.raises(UnsupportedTransitionError):
        build_transition_operator([[transition]], [env], (12,))
    operator = build_transition_operator(
        [[transition]], [env], (12,), allow_dense_fallback=True
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), np.eye(12))


def _local_and_two_environments():
    """A one-bin Local state in A and spatial states in A (25 bins) and B
    (15 bins), each with exterior bins."""
    envs = [environment((25,), "A"), environment((15,), "B")]
    observations = [
        ObservationModel("A", is_local=True),
        ObservationModel("A"),
        ObservationModel("B"),
    ]
    return envs, observations, (1, 25, 15)


@pytest.mark.unit
@pytest.mark.parametrize(
    "transitions",
    [
        [
            [Discrete(), Uniform("A", "A"), Uniform("A", "B")],
            [Discrete(), RandomWalk("A"), Uniform("A", "B")],
            [Discrete(), Uniform("B", "A"), RandomWalk("B")],
        ],
        [
            [Discrete(), RandomWalk("A"), Uniform("B")],
            [Uniform("A"), Identity("A"), Uniform("A", "B")],
            [Uniform("B"), Uniform("B", "A"), Uniform("B")],
        ],
    ],
    ids=["cross-environment-uniform", "same-environment-and-scalar-sources"],
)
@pytest.mark.parametrize("allow_dense_fallback", [False, True])
def test_one_bin_blocks_match_dense_across_environments(
    transitions, allow_dense_fallback
):
    """Blocks from and into a one-bin state are the same dense and
    structured: from it, uniform over the interior of the destination
    state's environment; into it, probability 1."""
    envs, observations, sizes = _local_and_two_environments()
    dense, state_ind = dense_reference(transitions, envs, sizes, observations)
    operator = build_transition_operator(
        transitions,
        envs,
        sizes,
        observation_models=observations,
        allow_dense_fallback=allow_dense_fallback,
    )
    np.testing.assert_allclose(
        np.asarray(LazyDenseTransition(operator)), dense, rtol=1e-12, atol=1e-15
    )
    local_row = dense[np.flatnonzero(state_ind == 0)[0]]
    for state, env in ((1, envs[0]), (2, envs[1])):
        interior = env.is_track_interior_flat_
        np.testing.assert_array_equal(
            local_row[state_ind == state], interior / interior.sum()
        )
    np.testing.assert_array_equal(dense[np.ix_(state_ind != 0, state_ind == 0)], 1.0)


@pytest.mark.unit
def test_discrete_from_a_one_bin_state_into_another_environment_is_rejected():
    """Discrete names no environment, so it cannot set a block from a
    one-bin state to a spatial state, dense or structured."""
    envs, observations, sizes = _local_and_two_environments()
    transitions = [
        [Discrete(), Uniform("A"), Discrete()],
        [Discrete(), RandomWalk("A"), Uniform("A", "B")],
        [Discrete(), Uniform("B", "A"), RandomWalk("B")],
    ]
    with pytest.raises(ValidationError, match="environment_name=None"):
        dense_reference(transitions, envs, sizes, observations)
    with pytest.raises(ValidationError, match="environment_name=None"):
        build_transition_operator(
            transitions, envs, sizes, observation_models=observations
        )


def _equal_size_environments():
    """Environments A and B with four bins each and disjoint interiors."""
    env_a, env_b = (
        environment((4,), "A", holes=False),
        environment((4,), "B", holes=False),
    )
    env_a.is_track_interior_[:] = [True, True, False, False]
    env_b.is_track_interior_[:] = [False, False, True, True]
    return [env_a, env_b]


class _CustomInA:
    environment_name = "A"

    def make_state_transition(self, environments, context=None):
        msg = "a one-bin block is not built by the transition"
        raise AssertionError(msg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "transition",
    [
        Uniform("A"),
        Uniform("A", "A"),
        Uniform("A", "typo"),
        Uniform("B", "A"),
        RandomWalk("A"),
        EmpiricalMovement("A"),
        _CustomInA(),
    ],
    ids=[
        "uniform-source-only",
        "uniform-a-to-a",
        "uniform-typo",
        "uniform-b-to-a",
        "random-walk",
        "empirical",
        "custom",
    ],
)
def test_one_bin_block_into_another_environment_must_declare_it(transition):
    """From a one-bin state in A into a state in B, the transition must
    declare B as its destination; dense and structured raise the same
    error."""
    envs = _equal_size_environments()
    observations = [ObservationModel("A", is_local=True), ObservationModel("B")]
    transitions = [[Discrete(), transition], [Discrete(), Identity("B")]]
    with pytest.raises(ValidationError) as dense_error:
        dense_reference(transitions, envs, (1, 4), observations)
    with pytest.raises(ValidationError) as structured_error:
        build_transition_operator(
            transitions, envs, (1, 4), observation_models=observations
        )
    assert str(dense_error.value) == str(structured_error.value)


@pytest.mark.unit
def test_random_walk_into_a_smaller_environment_must_declare_it():
    """RandomWalk("A") from a one-bin state into a 3-bin B state is rejected
    by name, not built over B."""
    envs = [environment((5,), "A"), environment((3,), "B", holes=False)]
    observations = [ObservationModel("A", is_local=True), ObservationModel("B")]
    transitions = [[Discrete(), RandomWalk("A")], [Discrete(), Identity("B")]]
    match = r"RandomWalk declares destination environment 'A'[\s\S]*'B'"
    with pytest.raises(ValidationError, match=match):
        dense_reference(transitions, envs, (1, 3), observations)
    with pytest.raises(ValidationError, match=match):
        build_transition_operator(
            transitions, envs, (1, 3), observation_models=observations
        )


@pytest.mark.unit
def test_spatial_block_must_declare_its_source_environment():
    """A spatial block between two B states built from A's environment is
    rejected even when A and B have the same number of bins."""
    envs = _equal_size_environments()
    observations = [ObservationModel("B"), ObservationModel("B")]
    transitions = [[Identity("A"), Uniform("B")], [Uniform("B"), Identity("B")]]
    match = r"Identity declares source environment 'A'[\s\S]*'B'"
    with pytest.raises(ValidationError, match=match):
        dense_reference(transitions, envs, (4, 4), observations)
    with pytest.raises(ValidationError, match=match):
        build_transition_operator(
            transitions, envs, (4, 4), observation_models=observations
        )


@pytest.mark.unit
def test_one_bin_block_uses_the_declared_destination_without_observation_models():
    """Called without observation models, build_transition_operator takes a
    one-bin block's destination from the transition's declared destination,
    as the dense constructor does with consistent observation models."""
    envs = _equal_size_environments()
    observations = [ObservationModel("A", is_local=True), ObservationModel("B")]
    transitions = [[Discrete(), Uniform("A", "B")], [Discrete(), Identity("B")]]
    dense, _ = dense_reference(transitions, envs, (1, 4), observations)
    np.testing.assert_array_equal(dense[0, 1:], [0.0, 0.0, 0.5, 0.5])
    operator = build_transition_operator(transitions, envs, (1, 4))
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)
