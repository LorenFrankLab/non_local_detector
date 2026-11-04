"""The allow-list of formula syntax that may reach patsy.

patsy evaluates each term of a formula as Python, so a transition formula is
checked against a small grammar before patsy sees it. These tests pin which
formulas pass (every formula the package, its tests, documentation and
notebooks use) and that code-execution tricks are refused before evaluation.
"""

import numpy as np
import pytest
from patsy import dmatrix

from non_local_detector._formula_grammar import check_design_info, check_formula
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
    "1 + I(speed ** 2) + I(speed ^ phase) + I(-speed) + I(+speed)",
    "1 + I(speed / 10.0 - 2 * phase)",
    "1 + bs(speed, df=5, degree=3, include_intercept=False)",
    "1 + bs(speed, df=5, lower_bound=0, upper_bound=None)",
    "1 + bs(speed, knots=(1, 2, -3.5))",
    "1 + cr(speed, df=5) + cc(phase, df=4)",
    "1 + te(cr(speed, df=5), cc(phase, df=5))",
    "1 + center(speed) + standardize(phase)",
    "1 + C(arm)",
    "1 + np.log(speed) + np.exp(speed) + np.sqrt(speed) + np.abs(speed)",
    # patsy reads a formula without a response the same with or without "~".
    "~ speed",
]


@pytest.mark.parametrize("formula", DOCUMENTED_FORMULAS + GRAMMAR_FORMULAS)
def test_allowed_formulas_pass(formula):
    check_formula(formula)


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
    "C(arm, Treatment('a'))",
    "cr(speed, constraints='center')",
    "I(f'{speed}')",
    "I('{0.__class__}'.format(speed))",
    "I(b'speed')",
    "I('%s' % speed)",
    # Operators outside the grammar.
    "I(speed % 2)",
    "I(speed // 2)",
    "I(speed @ phase)",
    "I(speed | phase)",
    "speed | phase",
    "I(~speed)",
    "I(not speed)",
    "I(speed and phase)",
    "I(1j)",
    "I(...)",
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


def test_check_does_not_evaluate_the_formula(monkeypatch):
    """Checking parses the formula but never evaluates it."""
    calls = []
    monkeypatch.setattr("builtins.eval", lambda *args: calls.append(args))
    with pytest.raises(ValidationError):
        check_formula("__import__('os').system('x')")
    check_formula("1 + quantile_bs(speed)")
    assert calls == []


def test_design_info_of_an_allowed_formula_passes():
    speed = np.linspace(0.0, 10.0, 50)
    design = dmatrix("1 + bs(speed, df=4) + np.log1p(speed)", {"speed": speed})
    check_design_info(design.design_info)


def test_design_info_whose_factor_code_is_refused_fails():
    """A design whose stored factor code would run something else is refused
    before it is evaluated on new covariates."""
    speed = np.linspace(0.0, 10.0, 50)
    design = dmatrix("1 + I(speed)", {"speed": speed})
    (factor,) = (factor for factor in design.design_info.factor_infos)
    factor.code = "__import__('os').system('x')"
    with pytest.raises(ValidationError, match="not allowed"):
        check_design_info(design.design_info)
