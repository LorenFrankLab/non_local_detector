"""Exact structured products against the unchanged dense model constructor."""

import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from non_local_detector.continuous_state_transitions import (
    Discrete,
    EmpiricalMovement,
    Identity,
    RandomWalk,
    RandomWalkDirection1,
    Uniform,
)
from non_local_detector.environment import Environment
from non_local_detector.graph_distances import LazyGraphDistances
from non_local_detector.models._defaults import _ModelDefaults
from non_local_detector.models.base import _DetectorBase
from non_local_detector.observation_models import ObservationModel
from non_local_detector.transition_operators import (
    DenseTransitionBudgetError,
    LazyDenseTransition,
    UnsupportedTransitionError,
    build_transition_operator,
)


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
    eager = Environment(
        **kwargs, is_track_interior=mask, is_track_interior_=mask
    ).fit_place_grid(position, infer_track_interior=False)
    lazy = Environment(
        **kwargs, is_track_interior=mask, is_track_interior_=mask
    ).fit_place_grid(
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
    assert descriptor.environment is lazy
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
        Environment(
            **kwargs, is_track_interior=mask, is_track_interior_=mask
        ).fit_place_grid(
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
        raise AssertionError("allocated an invalid spatial dense block")

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
        def make_state_transition(self, environments):
            raise AssertionError(
                "opaque constructor cannot reinterpret sparse distances"
            )

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

        def make_state_transition(self, environments):
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

        def make_state_transition(self, environments):
            self.environment = environments[0]
            raise AssertionError("opaque lazy fallback must reject before construction")

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
    model._get_environment_by_name = lambda name: next(
        env for env in envs if env.environment_name == name
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
    with enable_x64(dtype == np.float64):
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
    with enable_x64(dtype == np.float64):
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
    with enable_x64(dtype == np.float64):
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


@pytest.mark.unit
def test_rectangular_multienvironment_uniform_blocks():
    env1, env2 = environment((4, 3), "one"), environment((3, 5), "two")
    transitions = [
        [Identity("one"), Uniform("one", "two")],
        [Uniform("two", "one"), Identity("two")],
    ]
    sizes = (12, 15)
    dense, _ = dense_reference(transitions, [env1, env2], sizes)
    operator = build_transition_operator(transitions, [env1, env2], sizes)
    assert_products(operator, dense, np.float32)


@pytest.mark.unit
def test_scalar_source_uses_original_environment_name_even_with_second_environment():
    env1, env2 = environment((5,), "one"), environment((5,), "two")
    env1.is_track_interior_[:] = [True, False, True, True, False]
    env2.is_track_interior_[:] = [False, True, False, False, True]
    transitions = [[Discrete(), Uniform("one", "two")], [Discrete(), Identity("two")]]
    dense, _ = dense_reference(transitions, [env1, env2], (1, 5))
    operator = build_transition_operator(transitions, [env1, env2], (1, 5))
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
    with enable_x64(True):
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
    env.position_range = ((-0.5, 4.5), (-0.5, 4.5))
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
        transitions, [env], (5,), allow_dense_fallback=True, **context
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)
    # Base defaults absent labels to group zero before calling the empirical
    # estimator, rather than its standalone "all groups" default.
    transitions = [[EmpiricalMovement(encoding_group=1)]]
    dense, _ = dense_reference(transitions, [env], (5,), position=position)
    operator = build_transition_operator(
        transitions, [env], (5,), allow_dense_fallback=True, position=position
    )
    np.testing.assert_array_equal(np.asarray(LazyDenseTransition(operator)), dense)


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

    with enable_x64(dtype == np.float64):
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
            expected_final, (filtered, predicted) = _filter_covariate_dependent_impl(
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
            expected_final, (filtered, predicted) = _filter_impl(
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

    with enable_x64(True):
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

    with enable_x64(True):
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
