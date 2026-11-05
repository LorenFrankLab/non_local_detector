"""Saving and loading of detectors and their results.

``_DetectorBase`` inherits ``save_model``, ``load_model``, ``save_results``
and ``load_results`` from ``_PersistenceMixin``. ``_ModelUnpickler`` loads
detectors pickled by earlier versions, and ``_upgrade_saved_state`` fills in
what a detector saved by an earlier version lacks; ``_DetectorBase.__setstate__``
calls it.

The detector classes keep their import paths in ``models.base``, which old
pickles name, and every name moved here stays importable from there.
"""

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
import xarray as xr

import non_local_detector  # resolves load_model's return annotation
from non_local_detector._formula_grammar import UntrustedFormula
from non_local_detector.discrete_state_transitions import (
    _row_smoothness,
    _smoothness_matrix,
)
from non_local_detector.likelihoods import (
    _record_legacy_encoding_algorithm,
    _unify_stored_encoding_values,
)
from non_local_detector.models._results import label_checkpointed_result
from non_local_detector.result_store import DEFAULT_MAX_READ_BYTES

if TYPE_CHECKING:
    import non_local_detector.models.base
    from non_local_detector.models.base import _DetectorBase


# Dataset attribute in which ``save_results`` records integer dtypes.
_INTEGER_DTYPES_ATTR = "non_local_detector_integer_dtypes"


# Constructor parameters added in 2026-04, with the value that reproduces how
# a detector saved before then was fitted: the fixed-count Dirichlet prior,
# no frozen discrete-transition rows (No-Spike was re-estimated too), and the
# single-bin Local state.
_PARAMETERS_BEFORE_THEY_EXISTED: dict[str, object] = {
    "discrete_transition_prior_weight": 0.0,
    "frozen_discrete_transition_rows": None,
    "local_position_std": None,
}


class _ModelUnpickler(pickle.Unpickler):
    """Unpickle a detector, including one saved with JAX 0.4.30 or earlier.

    Those JAX releases pickled each array's abstract value with a
    ``named_shape`` entry, which later releases no longer accept, so the
    detectors of the non_local_detector 0.6 releases that resolved to them
    raised ``TypeError`` on load. The entry is dropped; JAX no longer has
    named shapes.
    """

    def find_class(self, module: str, name: str) -> Any:
        found = super().find_class(module, name)
        if (module, name) != ("jax._src.array", "_reconstruct_array"):
            return found

        def reconstruct_array(*args: Any, **kwargs: Any) -> jax.Array:
            # (fun, args, arr_state, aval_state) in the JAX releases so far;
            # anything else passes through unchanged.
            if len(args) == 4 and isinstance(args[3], dict):
                aval_state = {k: v for k, v in args[3].items() if k != "named_shape"}
                args = (*args[:3], aval_state)
            return cast("jax.Array", found(*args, **kwargs))

        return reconstruct_array


def _upgrade_saved_state(detector: "_DetectorBase", state: dict[str, Any]) -> None:
    """Fill in what a detector unpickled from ``state`` lacks.

    ``_DetectorBase.__setstate__`` calls this after restoring ``state``, so
    detectors saved by earlier versions load with the attributes this
    version reads.

    Parameters
    ----------
    detector : _DetectorBase
        The detector being unpickled, with ``state`` restored.
    state : dict
        The pickled attributes.
    """
    # Constructor parameters added after a detector was saved: the values
    # under which it was fitted, so it prints and reports get_params.
    for name, value in _PARAMETERS_BEFORE_THEY_EXISTED.items():
        if name not in state:
            setattr(detector, name, value)
    # Detectors saved after a prediction carry its degenerate bins, which
    # decoding used to record on the detector; they describe that
    # recording, not the model.
    detector.__dict__.pop("_degenerate_timesteps_", None)
    # Earlier versions cached the environment lookup on the detector;
    # nothing reads it now.
    detector.__dict__.pop("_env_lookup_cache_", None)
    covariate_data = state.get("_discrete_transition_covariate_data_")
    # A formula a saved model file was not trusted with stays unevaluated:
    # the copy is for inspection only, as the loaded model is.
    untrusted = isinstance(
        getattr(detector.discrete_transition_type, "formula", None), UntrustedFormula
    )
    if (
        covariate_data is not None
        and detector.discrete_transition_design_matrix_ is None
        and not untrusted
    ):
        detector.discrete_transition_design_matrix_ = (
            detector.discrete_transition_type.make_state_transition(
                covariate_data
            ).design_matrix
        )
    # Detectors fitted before transition smoothing existed: no penalty
    # strengths and no covariance, so EM and intervals see an unsmoothed
    # fit that has not estimated its uncertainty.
    if "discrete_transition_coefficient_covariance_" not in state:
        detector.discrete_transition_coefficient_covariance_ = None
    if "discrete_transition_smoothness_matrix_" not in state:
        design = getattr(detector, "discrete_transition_design_matrix_", None)
        if design is None:
            detector.discrete_transition_smoothness_matrix_ = None
            detector.discrete_transition_smoothness_ = None
        else:
            penalties = _smoothness_matrix(design.design_info)
            detector.discrete_transition_smoothness_matrix_ = penalties
            # The design matrix is only set together with the coefficients.
            assert detector.discrete_transition_coefficients_ is not None
            detector.discrete_transition_smoothness_ = _row_smoothness(
                0.0,
                None,
                penalties,
                detector.discrete_transition_coefficients_.shape[1],
            )[0]
    # Detectors fitted before ``environments_`` existed kept the fitted
    # environments in ``environments``. They stay there, where code that
    # inspects old models reads them, and ``environments_`` is the same
    # tuple; only a new fit splits configuration from fitted copies. Those
    # releases stored a list, which the constructor (and so ``clone``)
    # turns into a tuple.
    if "environments_" not in state and any(
        env._is_fitted for env in detector.environments
    ):
        detector.environments = tuple(detector.environments)
        detector.environments_ = detector.environments
    # Encoding models fitted before each recorded its algorithm, or before
    # the shared keys' values were NumPy arrays.
    encoding_model = state.get("encoding_model_")
    if isinstance(encoding_model, Mapping):
        for entry in encoding_model.values():
            _record_legacy_encoding_algorithm(entry)
            _unify_stored_encoding_values(entry)


class _PersistenceMixin:
    """Saving and loading methods of ``_DetectorBase``."""

    def save_model(self, filename: str | Path = "model.nld") -> None:
        """
        Save the detector to a file.

        The file is a zip of JSON documents (the configuration and the
        fitted state) and ``.npy`` arrays, which ``load_model`` reads in this
        and every later version of non_local_detector, and which can be read
        without it (see the "Saving and loading models" documentation). It
        stores what the fit learned, and the inputs of the structures it
        rebuilds on load (environment grids, the continuous transition and
        any covariate design), not the structures themselves. ``.nld`` is the
        conventional suffix.

        Parameters
        ----------
        filename : str or Path, optional
            Where to write the model, by default "model.nld". An existing file
            is replaced.

        Raises
        ------
        ValidationError
            If the detector, or an object its parameters hold (a custom
            transition, for example), is of a type that is not registered for
            saving (see ``non_local_detector.model_format.register_type``), or
            it is a loaded detector whose rebuilt structures no longer match
            the saved ones.
        NotFittedError
            If the detector is only partly fitted.
        """
        from non_local_detector.model_format._detector import save_detector

        save_detector(cast("_DetectorBase", self), filename)

    @staticmethod
    def load_model(
        filename: str | Path = "model.nld",
        *,
        trust_formulas: bool = False,
    ) -> "non_local_detector.models.base._DetectorBase":
        """
        Load a detector saved with ``save_model``, or pickled by earlier releases.

        The format is detected from the file's first bytes, not its suffix.
        A file in the saved-model format is read without running code from
        it, each array decompressed straight into place, and is closed before
        this returns. Its environment grids, continuous transition and
        covariate design are rebuilt; before the loaded detector first
        decodes, they are compared with fingerprints saved with the model,
        and if they differ (because the code that rebuilds them changed)
        decoding raises ``ValidationError`` asking for a refit. Inspecting the
        model always works. A file written in a newer format version is
        refused, naming the version to upgrade to.

        patsy evaluates a covariate transition's formula as Python, and a
        saved file may be someone else's. Unless ``trust_formulas`` is True,
        a formula outside the formula grammar for files (covariate names,
        numbers, arithmetic, the package's spline transforms, elementwise
        numpy functions such as ``np.log`` and ``np.sin``, and ``C()`` with
        patsy's contrasts) is not evaluated: the model loads for inspection
        only, with a ``ParameterWarning`` naming the formula, and its
        transition's ``formula`` is an inert ``UntrustedFormula`` that
        copies, pickles and ``sklearn.clone`` carry. Decoding, fitting,
        transition intervals, transition plots and ``save_model`` of the
        model or any copy raise ``ValidationError`` until the transition is
        given a formula of your own. A formula within the grammar is a
        ``LoadedFormula`` (a ``str``), checked again with the covariate names
        when a design is built from it. By default loading never runs code
        from a ``.nld`` file, but a crafted file can still make loading or
        decoding slow or memory-hungry (e.g. very large spline
        interactions), so load files from untrusted sources with care.

        A pickle saved by an earlier release loads for inspection, including
        one pickled with JAX 0.4.30 or earlier. Decoding with one fitted
        before rates were fitted in Hz (the 0.6 releases) raises
        ``ValidationError`` asking for a refit. ``convert_model`` rewrites a
        pickle that can decode in the saved-model format.

        .. warning::

            Only load pickles from trusted sources. Pickle files can execute
            arbitrary code during deserialization, so a pickle's formula is
            not checked and ``trust_formulas`` does not apply to it.

        Parameters
        ----------
        filename : str or Path, optional
            Filename to load the model from, by default "model.nld".
        trust_formulas : bool, optional
            Evaluate a saved model's covariate transition formula even if it
            is outside the formula grammar, by default False. Pass True only
            for a file you trust, such as one you saved yourself.

        Returns
        -------
        _DetectorBase
            Loaded detector instance

        Raises
        ------
        ValidationError
            If the file names a type not registered in this session (import
            the package that registers it first), or was written in a newer
            format version.
        DataError
            If an array of the file does not match its checksum (a corrupt or
            edited file).
        """
        from non_local_detector.model_format._container import is_saved_model_format

        if is_saved_model_format(filename):
            from non_local_detector.model_format._detector import load_detector

            return load_detector(filename, trust_formulas=trust_formulas)
        with Path(filename).open("rb") as f:
            return cast("_DetectorBase", _ModelUnpickler(f).load())

    @staticmethod
    def save_results(results: xr.Dataset, filename: str = "results.nc") -> None:
        """
        Save the results to a netCDF file.

        ``state_bins`` is a MultiIndex, which netCDF does not support, so it
        is converted before saving. Boolean attributes are stored as integers
        and the internal ``_native_metadata`` attribute is dropped. The dtype
        of each integer variable and coordinate is recorded so that
        ``load_results`` restores it: netCDF3, written when neither netCDF4
        nor h5netcdf is installed, stores int64 as int32.

        Parameters
        ----------
        results : xr.Dataset
            Decoding results.
        filename : str, optional
            Name to save, by default "results.nc".
        """
        export = (
            results.reset_index("state_bins")
            if "state_bins" in results.indexes
            else results.copy(deep=False)
        )
        export.attrs = {
            name: int(value) if isinstance(value, (bool, np.bool_)) else value
            for name, value in export.attrs.items()
            if name != "_native_metadata"
        }
        export.attrs[_INTEGER_DTYPES_ATTR] = json.dumps(
            {
                str(name): variable.dtype.name
                for name, variable in export.variables.items()
                if variable.dtype.kind in "iu"
            }
        )
        export.to_netcdf(filename)

    @staticmethod
    def load_results(
        filename: str | Path = "results.nc",
        *,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> xr.Dataset:
        """
        Load results saved by ``save_results`` or a checkpointed result_path.

        A netCDF file is read lazily and gets its ``state_bins`` MultiIndex
        back and its integer variables and coordinates the dtypes they were
        saved with; closing the returned dataset closes the file. A directory
        is opened as a checkpointed result store with lazy arrays.

        Parameters
        ----------
        filename : str or Path, optional
            A netCDF file or checkpointed result directory, by default
            "results.nc".
        max_read_bytes : int, optional
            Maximum bytes in one explicit read of a result directory's lazy
            arrays, by default 512 MiB. Ignored for netCDF files.

        Returns
        -------
        xr.Dataset
            Decoding results
        """
        if Path(filename).is_dir():
            from non_local_detector.result_store import open_result_store

            results = open_result_store(filename, max_read_bytes=max_read_bytes)
            metadata = results.attrs.get("_native_metadata")
            if metadata:
                metadata = {**metadata, "max_read_bytes": max_read_bytes}
                return label_checkpointed_result(results, metadata)
            return results
        opened = xr.open_dataset(filename)
        results = opened
        if _INTEGER_DTYPES_ATTR in results.attrs:
            integer_dtypes = json.loads(results.attrs.pop(_INTEGER_DTYPES_ATTR))
        else:
            # Files saved before the dtypes were recorded: restore state_ind.
            integer_dtypes = {"state_ind": np.dtype(int).name}
        # Only a still-integer variable is restored: one edited to another
        # dtype after saving (say, float with NaN) keeps its values.
        restored = {
            name: results.variables[name].astype(dtype)
            for name, dtype in integer_dtypes.items()
            if name in results.variables
            and results.variables[name].dtype.kind in "iu"
            and results.variables[name].dtype != dtype
        }
        results = results.assign_coords(
            {name: value for name, value in restored.items() if name in results.coords}
        ).assign(
            {
                name: value
                for name, value in restored.items()
                if name in results.data_vars
            }
        )
        if "state_bins" in results.dims:
            if "state_ind" in results.dims:
                # Earlier versions saved state_ind on a dimension of its own.
                results = results.drop_vars("state_ind").assign_coords(
                    state_ind=("state_bins", results["state_ind"].values)
                )
            # Scalar state/environment coordinates are also attached to this
            # array, but only the original one-dimensional levels form its
            # spatial index; state_ind is a plain coordinate along it.
            coord_names = [
                name
                for name, coordinate in results.coords.items()
                if name not in {"state_bins", "state_ind"}
                and coordinate.dims == ("state_bins",)
            ]
            results = results.set_index(state_bins=coord_names)
        # The transformations above drop the file's close handle; closing the
        # returned results (or leaving a ``with`` block) must close the file.
        results.set_close(opened.close)
        return results
