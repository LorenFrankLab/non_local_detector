"""The allow-list of syntax a covariate-transition formula may use.

patsy evaluates every term of a formula as Python, so a formula such as
``"__import__('os').system(...)"`` would run code when a design matrix is
built. A transition formula may come from a saved model, so it is checked
against a small grammar before patsy evaluates it.

The check uses patsy's own parser (``ModelDesc.from_formula``), which splits a
formula at its patsy operators (``+ - * / : ** ~`` and parentheses) without
evaluating anything, into the Python expressions patsy later evaluates: the
``code`` of each term's factors. Each of these is parsed with :mod:`ast` and
accepted only if every node is one of

- a covariate name: an identifier that is not called and does not start with
  ``__``;
- a number, ``True``, ``False`` or ``None``;
- the operators ``+ - * / ** ^`` and unary ``+`` and ``-``;
- a list or tuple of these (e.g. ``knots=[1.0, 4.0]``);
- ``np.pi``;
- a call of one of the documented transforms (``_FUNCTIONS``) or numpy
  functions (``_NUMPY_FUNCTIONS``), with arguments, and keyword arguments,
  that are themselves allowed.

Everything else is refused: other attribute access, subscripts, strings,
lambdas, comprehensions, conditionals, comparisons, and calls of anything
else. Validating the factor code patsy itself extracted means the check sees
exactly the Python that patsy would evaluate, so a string cannot be split one
way for the check and another way for evaluation.
"""

import ast
from typing import NoReturn

from patsy import ModelDesc, PatsyError
from patsy.design_info import DesignInfo

from non_local_detector.exceptions import ValidationError

# The package's spline and transform functions: patsy's built-in ``bs``,
# ``cr``, ``cc``, ``te``, ``center``, ``standardize``, ``C`` and ``I``, and
# the package's ``quantile_bs`` and ``cyclic_bs``.
_FUNCTIONS = frozenset(
    {
        "C",
        "I",
        "bs",
        "cc",
        "center",
        "cr",
        "cyclic_bs",
        "quantile_bs",
        "standardize",
        "te",
    }
)
# Called as ``np.<name>(...)``.
_NUMPY_FUNCTIONS = frozenset({"abs", "exp", "log", "log1p", "sqrt"})
# Read as ``np.<name>``, e.g. ``upper=2 * np.pi``.
_NUMPY_CONSTANTS = frozenset({"pi"})
_BINARY_OPERATORS: tuple[type[ast.operator], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.BitXor,
)
_UNARY_OPERATORS: tuple[type[ast.unaryop], ...] = (ast.UAdd, ast.USub)

_ALLOWED = (
    "covariate names, numbers, True, False and None, the operators "
    "+ - * / : ** ^ and parentheses, lists or tuples of numbers, np.pi, and "
    "calls of "
    + ", ".join(sorted(_FUNCTIONS, key=str.lower))
    + ", "
    + ", ".join(f"np.{name}" for name in sorted(_NUMPY_FUNCTIONS))
)
_HINT = (
    "patsy evaluates a formula as Python, so transition formulas are limited "
    "to this grammar. Compute any other transform of a covariate before "
    "fitting and pass it as its own covariate, e.g. "
    "covariate_data['log_speed'] = np.log(speed) with formula "
    "'1 + quantile_bs(log_speed)'."
)


def check_formula(formula: object) -> None:
    """Check that a transition formula only uses the allowed grammar.

    Parameters
    ----------
    formula : str
        Right-hand-side patsy formula, e.g. ``"1 + quantile_bs(speed)"``.
        Typed ``object`` because a stored or user-set formula may be of any
        type, and anything but a string is refused.

    Raises
    ------
    ValidationError
        If ``formula`` is not a string, cannot be parsed, has a response
        (left of ``~``), or uses syntax outside the grammar; the message
        names the offending element and the allowed set.
    """
    if not isinstance(formula, str):
        msg = "A transition formula must be a string."
        raise ValidationError(
            msg,
            expected='a patsy formula, e.g. "1 + quantile_bs(speed)"',
            got=type(formula).__name__,
        )
    try:
        description = ModelDesc.from_formula(formula)
    except PatsyError as error:
        msg = f"Transition formula {formula!r} could not be parsed: {error}"
        raise ValidationError(
            msg,
            expected=f"a patsy formula using {_ALLOWED}",
            got=repr(formula),
            hint=_HINT,
        ) from None
    if description.lhs_termlist:
        _refuse(formula, "a response (left of '~')")
    for term in description.rhs_termlist:
        for factor in term.factors:
            _check_expression(factor.name(), formula)


def check_design_info(design_info: DesignInfo) -> None:
    """Check the Python a patsy design evaluates on new covariates.

    A design built from a formula re-evaluates the code of each of its
    factors whenever it is applied to new data (``build_design_matrices``).

    Parameters
    ----------
    design_info : patsy.DesignInfo
        ``design_info`` of a transition design matrix.

    Raises
    ------
    ValidationError
        If the code of a factor uses syntax outside the grammar.
    """
    formula = design_info.describe()
    for factor in design_info.factor_infos:
        _check_expression(factor.name(), formula)


def _check_expression(code: str, formula: str) -> None:
    """Refuse a Python expression of ``formula`` outside the grammar."""
    try:
        expression = ast.parse(code, mode="eval").body
    except SyntaxError:
        _refuse(formula, f"the invalid Python expression {code!r}")
    offending = _offending(expression)
    if offending is not None:
        _refuse(formula, offending)


def _offending(node: ast.AST) -> str | None:
    """Description of the first node of ``node`` outside the grammar, or None."""
    if isinstance(node, ast.Name):
        if node.id.startswith("__"):
            return f"the name {node.id}"
        if node.id == "np":
            return "the bare name np"
        return None
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, bool | int | float):
            return None
        return f"the literal {ast.unparse(node)}"
    if isinstance(node, ast.Attribute):
        if _numpy_member(node) in _NUMPY_CONSTANTS:
            return None
        return f"the attribute access {ast.unparse(node)}"
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BINARY_OPERATORS):
            return f"the operator in {ast.unparse(node)}"
        return _offending(node.left) or _offending(node.right)
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARY_OPERATORS):
            return f"the operator in {ast.unparse(node)}"
        return _offending(node.operand)
    if isinstance(node, ast.List | ast.Tuple):
        return _first_offending(node.elts)
    if isinstance(node, ast.Call):
        return _offending_call(node)
    return f"the {type(node).__name__} expression {ast.unparse(node)}"


def _offending_call(call: ast.Call) -> str | None:
    """Description of what a call does outside the grammar, or None."""
    function = call.func
    if isinstance(function, ast.Name):
        allowed = function.id in _FUNCTIONS
    else:
        allowed = (
            isinstance(function, ast.Attribute)
            and _numpy_member(function) in _NUMPY_FUNCTIONS
        )
    if not allowed:
        return f"the call of {ast.unparse(function)} in {ast.unparse(call)}"
    for keyword in call.keywords:
        if keyword.arg is None:
            return f"the ** argument in {ast.unparse(call)}"
    return _first_offending([*call.args, *(keyword.value for keyword in call.keywords)])


def _first_offending(nodes: list[ast.expr]) -> str | None:
    """:func:`_offending` of the first of ``nodes`` outside the grammar."""
    for node in nodes:
        offending = _offending(node)
        if offending is not None:
            return offending
    return None


def _numpy_member(node: ast.Attribute) -> str | None:
    """``name`` of ``np.<name>``, or None for any other attribute access."""
    if isinstance(node.value, ast.Name) and node.value.id == "np":
        return node.attr
    return None


def _refuse(formula: str, offending: str) -> NoReturn:
    """Raise the error that ``formula`` uses ``offending``."""
    msg = f"Transition formula {formula!r} uses {offending}, which is not allowed."
    raise ValidationError(msg, expected=_ALLOWED, got=offending, hint=_HINT)
