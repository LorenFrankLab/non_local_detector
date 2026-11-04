"""The grammar a transition formula read from a saved model file must use.

patsy evaluates each term of a formula as Python, so ``load_model`` checks a
saved model's transition formula against a grammar before patsy sees it
(unless the file is trusted). These tests pin which formulas pass (every
formula the package, its tests, documentation and notebooks use, and common
elementwise transforms and contrasts) and that code-execution tricks are
refused before evaluation.
"""

import time

import numpy as np
import pytest
from patsy import dmatrix

from non_local_detector._formula_grammar import check_formula
from non_local_detector.exceptions import ValidationError

pytestmark = pytest.mark.unit

PERIOD = repr(2.0 * np.pi)

# Every transition formula the package, its tests, docs and notebooks use.
DOCUMENTED_FORMULAS = [
    "1 + quantile_bs(speed)",
    "0 + quantile_bs(speed)",
    "1 + speed",
    "1 + covariate",
    "1 + quantile_bs(x)",
    "1 + quantile_bs(speed, n_knots=5)",
    "1 + quantile_bs(speed, n_knots=8)",
    "1 + quantile_bs(covariate, n_knots=6)",
    "1 + quantile_bs(speed, upper_quantile=0.99)",
    "1 + quantile_bs(speed, n_knots=6, lower_quantile=0.05, upper_quantile=0.95)",
    "1 + quantile_bs(speed, lower_quantile=0.5, upper_quantile=0.5)",
    "1 + quantile_bs(speed, upper_quantile=1.5)",
    "1 + bs(speed, knots=[10.0, 20.0])",
    "1 + bs(speed, knots=[1.0, 4.0, 16.0])",
    "1 + np.log1p(speed) + quantile_bs(speed, n_knots=4)",
    "1 + cyclic_bs(head_direction, lower=0, upper=360)",
    f"1 + cyclic_bs(phase, lower=0.0, upper={PERIOD}, n_knots=8)",
    f"cyclic_bs(phase, lower=0.0, upper={PERIOD}, n_knots=30)",
    f"cyclic_bs(phase, lower=0.0, upper={PERIOD}, degree=1)",
    "cyclic_bs(phase, lower=1.0, upper=1.0)",
    "cyclic_bs(phase, lower=0.0, upper=6.0, n_knots=3)",
    "1 + quantile_bs(speed) + cyclic_bs(phase, lower=0.0, upper=6.283185307179586)",
    "1 + quantile_bs(speed) + cyclic_bs(phase, lower=0, upper=2 * np.pi)",
]

# Other syntax of the grammar: patsy operators, the documented transforms and
# the numpy functions.
GRAMMAR_FORMULAS = [
    "speed",
    "-1 + speed",
    "1 + speed - 1",
    "speed:phase",
    "speed * phase",
    "(speed + phase) ** 2",
    "speed / phase",
    "1 + I(speed ** 2) + I(speed ** -2) + I(speed ** 0.5) + I(speed ** 10)",
    "1 + I(-speed) + I(+speed)",
    "1 + I(speed / 10.0 - 2 * phase)",
    "1 + bs(speed, df=5, degree=3, include_intercept=False)",
    "1 + bs(speed, df=5, lower_bound=0, upper_bound=None)",
    "1 + bs(speed, knots=(1, 2, -3.5))",
    "1 + cr(speed, df=5) + cc(phase, df=4)",
    "1 + te(cr(speed, df=5), cc(phase, df=5))",
    "1 + center(speed) + standardize(phase)",
    "1 + C(arm)",
    "1 + C(arm, levels=[1, 2, 3])",
    "1 + np.log(speed) + np.exp(speed) + np.sqrt(speed) + np.abs(speed)",
    # patsy reads a formula without a response the same with or without "~".
    "~ speed",
]

# Elementwise numpy functions and C() contrasts, each of which patsy builds.
WIDENED_FORMULAS = [
    "1 + np.sin(phase) + np.cos(phase) + np.tan(phase)",
    "1 + np.arcsin(x) + np.arccos(x) + np.arctan(x) + np.arctan2(x, y)",
    "1 + np.log2(speed) + np.log10(speed) + np.expm1(x) + np.square(x)",
    "1 + np.sign(x) + np.floor(speed) + np.ceil(speed)",
    "1 + np.minimum(speed, 10) + np.maximum(speed, 0.5) + np.maximum(x, y)",
    "1 + C(arm, Treatment('b'))",
    '1 + C(arm, Treatment("b"))',
    "1 + C(arm, Treatment(reference='c'))",
    "1 + C(arm, contrast=Treatment('b'))",
    "1 + C(arm, Sum())",
    "1 + C(arm, Sum(omit='a'))",
    "1 + C(arm, Helmert())",
    "1 + C(arm, Diff())",
    "1 + C(arm, Poly())",
    "1 + C(arm, Poly(scores=[1, 2, 4]))",
    "1 + C(arm, levels=['c', 'b', 'a'])",
    "1 + C(arm, levels=('c', 'b', 'a'))",
    "1 + C(arm, Treatment('b'), levels=['c', 'b', 'a'])",
    "1 + C(arm, Treatment('b'), ['c', 'b', 'a'])",
    "1 + C(arm, Sum(), levels=['a', 'b', 'c']):np.sin(phase)",
]

COVARIATES = {
    "speed": np.linspace(1.0, 10.0, 6),
    "phase": np.linspace(0.0, 2 * np.pi, 6),
    "x": np.linspace(-0.9, 0.9, 6),
    "y": np.linspace(0.5, 1.5, 6),
    "arm": np.array(["a", "b", "c", "a", "b", "c"]),
}


@pytest.mark.parametrize(
    "formula", DOCUMENTED_FORMULAS + GRAMMAR_FORMULAS + WIDENED_FORMULAS
)
def test_allowed_formulas_pass(formula):
    check_formula(formula)


@pytest.mark.parametrize("formula", WIDENED_FORMULAS)
def test_widened_formulas_are_valid_patsy(formula):
    """Each added function and contrast is one patsy evaluates, and the check
    counts only data keys as covariates (not contrasts or string literals)."""
    check_formula(formula, covariate_names=COVARIATES)
    design = dmatrix(formula, COVARIATES)
    assert np.all(np.isfinite(design))


ATTACKS = [
    # Calls of builtins and anything not in the allow-list.
    "__import__('os').system('x')",
    "1 + __import__('os').system('x')",
    "open('f')",
    "eval('1')",
    "exec('1')",
    "getattr(np, 'log')(speed)",
    "dmatrix('1', speed)",
    "np.load('f')",
    "quantile_bs(speed)(speed)",
    # Python reads identifiers in NFKC form: these are open and __import__.
    "I(\uff4f\uff50\uff45\uff4e(speed))",
    "I(\uff3f\uff3fimport\uff3f\uff3f)",
    # Attribute access other than np.<allowed>.
    "np.__dict__",
    "().__class__",
    "I(().__class__)",
    "speed.__class__",
    "np.linalg.norm(speed)",
    "np.log.__self__",
    "np.log(speed).real",
    "quantile_bs.__globals__",
    "np.pi.real",
    "np.random",
    # Bare dunder names and numpy itself.
    "__builtins__",
    "1 + __name__",
    "np",
    # Subscripts and slices.
    "speed[0]",
    "speed[0:1]",
    "bs(speed, knots=knots[0])",
    # Lambdas, comprehensions, conditionals, comparisons, walrus, starred.
    "(lambda: 1)()",
    "I((lambda: 1)())",
    "I((lambda x: x)(speed))",
    "I([x for x in speed])",
    "I({x for x in speed})",
    "I({x: x for x in speed})",
    "I(sum(x for x in speed))",
    "I(speed if speed else phase)",
    "I(speed > 1)",
    "I((y := speed))",
    "bs(*speed)",
    "bs(speed, **options)",
    "I({'a': speed})",
    # Strings and format tricks.
    "'speed'",
    "cr(speed, constraints='center')",
    "I('a')",
    "np.sin('a')",
    "bs(speed, knots=['a'])",
    "Treatment('a')",
    "1 + C('arm')",
    "I(f'{speed}')",
    "I('{0.__class__}'.format(speed))",
    "I(b'speed')",
    "I('%s' % speed)",
    # Operators outside the grammar.
    "I(speed % 2)",
    "I(speed // 2)",
    "I(speed @ phase)",
    "I(speed | phase)",
    "I(speed ^ phase)",
    "speed | phase",
    "I(~speed)",
    "I(not speed)",
    "I(speed and phase)",
    "I(1j)",
    "I(...)",
    # C() calls a contrast given as its second argument: a bare name can be
    # any Python builtin, e.g. breakpoint, exit or copyright.
    "1 + C(speed, breakpoint)",
    "1 + C(speed, exit)",
    "1 + C(speed, contrast=copyright)",
    "1 + C(speed, Treatment)",
    # A contrast is a call of a patsy contrast with literal arguments only.
    "1 + C(arm, Treatment(breakpoint))",
    "1 + C(arm, Treatment(reference=exit))",
    "1 + C(arm, Treatment(open('f')))",
    "1 + C(arm, Treatment(np.log(2)))",
    "1 + C(arm, Treatment('a').__class__)",
    "1 + C(arm, Treatment('a')(arm))",
    "1 + C(arm, Treatment(*opts))",
    "1 + C(arm, Treatment(**opts))",
    "1 + C(arm, Treatment(f'{arm}'))",
    "1 + C(arm, Treatment(b'a'))",
    "1 + C(arm, Treatment([[1]]))",
    "1 + C(arm, Treatment(1j))",
    "1 + C(arm, np.log(arm))",
    "1 + C(arm, ContrastMatrix([[1]], ['a']))",
    "1 + C(arm, eval('1'))",
    # levels are a list or tuple of literals.
    "1 + C(arm, levels=[breakpoint])",
    "1 + C(arm, levels=['a', open('f')])",
    "1 + C(arm, levels='ab')",
    "1 + C(arm, levels=[['a']])",
    # C takes its data, a contrast and levels, nothing else.
    "1 + C(arm, Treatment('a'), ['a'], ['b'])",
    "1 + C(arm, Treatment('a'), contrast=Sum())",
    "1 + C(arm, other='a')",
    "1 + C(*arm)",
    # Other numpy functions.
    "np.sinh(speed)",
    "np.where(speed, 1, 0)",
    # Unbounded powers take unbounded time.
    "1 + I(speed*0 + 10**10**7)",
    "I(speed ** speed)",
    "I(speed ** 11)",
    "I(speed ** -11)",
    "I(speed ** (2 + 1))",
    "I((speed ** 2) ** 2)",
    "I((10 ** 10) ** 10)",
    # Lists hold numbers only.
    "bs(speed, knots=[a, b])",
    "bs(speed, knots=[np.log(2)])",
    # Line continuations and other control characters.
    "1 + \\\n speed",
    "1 +\nspeed",
    "1 + speed\r",
    "1 + speed\x00",
    # A response is not part of a transition formula.
    "y ~ speed",
    "np.log(y) ~ speed",
]


@pytest.mark.parametrize("formula", ATTACKS)
def test_attacks_are_refused(formula):
    # Some are refused by patsy's parser, the rest by the grammar.
    with pytest.raises(ValidationError, match=r"not allowed|could not be parsed"):
        check_formula(formula)


@pytest.mark.parametrize(
    "formula", ["speed +", "1 + bs(speed", "speed + 2", "lambda: 1"]
)
def test_unparseable_formulas_are_refused(formula):
    with pytest.raises(ValidationError, match="formula"):
        check_formula(formula)


@pytest.mark.parametrize("formula", [None, 1, b"1 + speed"])
def test_formula_must_be_a_string(formula):
    with pytest.raises(ValidationError, match="must be a string"):
        check_formula(formula)


def test_error_names_the_offending_element_and_the_allowed_set():
    with pytest.raises(ValidationError) as error:
        check_formula("1 + quantile_bs(speed) + open('f')")
    message = str(error.value)
    assert "open('f')" in message
    assert "the call of open" in message
    for name in ("quantile_bs", "cyclic_bs", "bs", "np.log1p", "np.pi"):
        assert name in message
    assert "Hint:" in message
    assert "trust_formulas=True" in message


@pytest.mark.parametrize(
    "formula",
    [
        "1 + exit",
        "1 + breakpoint",
        "1 + copyright + speed",
        "1 + bs(speed, knots=knots)",
        "1 + quantile_bs(speed, n_knots=covariate_data)",
        "1 + I(speed * velocity)",
        "1 + np.minimum(speed, cap)",
        "1 + C(arm, Treatment('a'))",
    ],
)
def test_names_must_be_covariates(formula):
    """patsy resolves a name it does not find in the data in the caller's
    namespace and Python's builtins, so every name must be a covariate."""
    with pytest.raises(ValidationError, match="not a covariate"):
        check_formula(formula, covariate_names=["speed", "phase"])


def test_covariate_names_pass():
    check_formula(
        "1 + quantile_bs(speed, n_knots=5) + I(speed * phase) + C(arm)",
        covariate_names=["speed", "phase", "arm"],
    )
    # Function names and np.pi are not covariates.
    check_formula(
        "1 + np.log1p(speed) + cyclic_bs(phase, lower=0, upper=2 * np.pi)",
        covariate_names=["speed", "phase"],
    )
    # Contrast names and string literals are not covariates either.
    check_formula(
        "1 + C(arm, Treatment('b'), levels=['a', 'b']) + np.sin(phase)",
        covariate_names=["arm", "phase"],
    )


def test_line_continuation_gives_a_validation_error():
    with pytest.raises(ValidationError, match="line break or backslash"):
        check_formula("1 + \\\n speed")


def test_check_does_not_evaluate_the_formula(monkeypatch):
    """Checking parses the formula but never evaluates it."""
    calls = []
    monkeypatch.setattr("builtins.eval", lambda *args: calls.append(args))
    with pytest.raises(ValidationError):
        check_formula("__import__('os').system('x')")
    check_formula("1 + quantile_bs(speed)")
    assert calls == []


NESTED_SQUARES = (
    "1 + speed + I(" + "np.square(" * 18 + "99999999999999999999" + ")" * 18 + ")"
)


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        # Python ints are unbounded and numpy keeps one beyond int64 as an
        # object, so each np.square would double its size.
        (NESTED_SQUARES, "int64"),
        ("1 + I(np.square(np.square(9999999999 * 9999999999)))", "int64"),
        ("1 + I(speed * (3037000500 * 3037000500))", "int64"),
        ("1 + bs(speed, knots=[9223372036854775808])", "int64"),
        ("1 + I(-9223372036854775808 + speed)", "int64"),
        ("1 + " + " + ".join(["speed"] * 400), "characters"),
        ("1 + I(" + "np.abs(" * 40 + "speed" + ")" * 40 + ")", "nested"),
    ],
    ids=[
        "nested-squares",
        "literal-product",
        "grouped-product",
        "list",
        "min",
        "long",
        "deep",
    ],
)
def test_bounded_formulas(formula, match):
    """A formula's evaluation time is bounded: integers stay within int64,
    and its length and nesting are limited."""
    start = time.perf_counter()
    with pytest.raises(ValidationError, match=match):
        check_formula(formula)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    "formula",
    [
        "1 + I(speed * 9223372036854775807)",
        "1 + I(speed * 4611686018427387904 + 2**10)",
        "1 + I(speed * 1e300 * 1e300)",
        "1 + I(" + "np.abs(" * 20 + "speed" + ")" * 20 + ")",
    ],
)
def test_bounds_admit_reasonable_formulas(formula):
    check_formula(formula)
