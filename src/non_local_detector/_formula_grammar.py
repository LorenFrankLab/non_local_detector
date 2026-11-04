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
- the operators ``+ - * /`` and unary ``+`` and ``-``, and ``**`` with an
  exponent that is a number of magnitude at most ``_MAX_EXPONENT`` and a base
  without another ``**`` (so evaluation time stays bounded);
- a list or tuple of numbers (e.g. ``knots=[1.0, 4.0]``);
- ``np.pi``;
- a call of one of the documented transforms (``_FUNCTIONS``) or numpy
  functions (``_NUMPY_FUNCTIONS``), with arguments, and keyword arguments,
  that are themselves allowed. ``C`` takes only its data and ``levels``:
  patsy calls a contrast given to it.

Everything else is refused: other attribute access, subscripts, strings,
lambdas, comprehensions, conditionals, comparisons, bitwise operators, line
breaks and backslashes, and calls of anything else. Validating the factor
code patsy itself extracted means the check sees exactly the Python that
patsy would evaluate, so a string cannot be split one way for the check and
another way for evaluation.

patsy looks up a name it does not find in the covariates in the caller's
namespace and Python's builtins (``breakpoint``, ``exit``, ...). Wherever the
covariates are known, every covariate name of the formula must therefore be
one of them (``covariate_names``).
"""

import ast
from collections.abc import Collection
from typing import NoReturn

from patsy import ModelDesc, PatsyError
from patsy.design_info import DesignInfo
from patsy.eval import replace_bare_funcalls

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
)
_UNARY_OPERATORS: tuple[type[ast.unaryop], ...] = (ast.UAdd, ast.USub)
# Largest magnitude of a ``**`` exponent.
_MAX_EXPONENT = 10

_ALLOWED = (
    "covariate names (keys of the covariate data), numbers, True, False and "
    "None, the operators + - * / : and parentheses, ** with a number of "
    f"magnitude at most {_MAX_EXPONENT} as exponent, lists or tuples of "
    "numbers, np.pi, and calls of "
    + ", ".join(sorted(_FUNCTIONS, key=str.lower))
    + ", "
    + ", ".join(f"np.{name}" for name in sorted(_NUMPY_FUNCTIONS))
    + " (C takes only its data and levels=)"
)
_HINT = (
    "patsy evaluates a formula as Python, so transition formulas are limited "
    "to this grammar. Compute any other transform of a covariate before "
    "fitting and pass it as its own covariate, e.g. "
    "covariate_data['log_speed'] = np.log(speed) with formula "
    "'1 + quantile_bs(log_speed)'."
)


def check_formula(
    formula: object, covariate_names: Collection[str] | None = None
) -> None:
    """Check that a transition formula only uses the allowed grammar.

    Parameters
    ----------
    formula : str
        Right-hand-side patsy formula, e.g. ``"1 + quantile_bs(speed)"``.
        Typed ``object`` because a stored or user-set formula may be of any
        type, and anything but a string is refused.
    covariate_names : collection of str, optional
        Keys of the covariate data the formula will be evaluated on. When
        given, every covariate name of the formula must be one of them.

    Raises
    ------
    ValidationError
        If ``formula`` is not a string, cannot be parsed, has a response
        (left of ``~``), uses syntax outside the grammar, or names a
        covariate missing from ``covariate_names``; the message names the
        offending element and the allowed set.
    """
    if not isinstance(formula, str):
        msg = "A transition formula must be a string."
        raise ValidationError(
            msg,
            expected='a patsy formula, e.g. "1 + quantile_bs(speed)"',
            got=type(formula).__name__,
        )
    # patsy's tokenizer fails with a bare AssertionError on some line
    # continuations, and no formula needs one.
    if "\\" in formula or not formula.replace("\t", " ").isprintable():
        _refuse(formula, "a line break or backslash or another control character")
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
            _check_expression(factor.name(), formula, covariate_names)


def check_design_info(
    design_info: DesignInfo, covariate_names: Collection[str] | None = None
) -> None:
    """Check the Python a patsy design evaluates on new covariates.

    ``build_design_matrices`` evaluates each factor's stored ``eval_code``:
    its ``code`` with the stateful transforms (``center``, ``bs``,
    ``quantile_bs``, ...) replaced by the objects fitted on the original
    data. The code is checked against the grammar, and the ``eval_code``
    must be exactly what patsy derives from it.

    Parameters
    ----------
    design_info : patsy.DesignInfo
        ``design_info`` of a transition design matrix.
    covariate_names : collection of str, optional
        Keys of the covariate data the design will be evaluated on. When
        given, every covariate name of the design must be one of them.

    Raises
    ------
    ValidationError
        If the code of a factor uses syntax outside the grammar or names a
        covariate missing from ``covariate_names``, or its ``eval_code`` is
        missing or differs from the code.
    """
    formula = design_info.describe()
    for factor, factor_info in design_info.factor_infos.items():
        code = factor.name()
        _check_expression(code, formula, covariate_names)
        state = factor_info.state
        transforms = state.get("transforms", {})
        if state.get("eval_code") != _eval_code(code, transforms):
            _refuse(
                formula,
                f"a stored eval_code {state.get('eval_code')!r} that is not "
                f"the code {code!r} of its factor",
            )


def _eval_code(code: str, transforms: Collection[str]) -> str:
    """Return the ``eval_code`` patsy derives from a factor's ``code``.

    patsy (``EvalFactor.memorize_passes_needed``) names the stateful
    transform called ``i``-th ``_patsy_stobj{i}__{name}__`` and replaces the
    call ``name(...)`` with ``_patsy_stobj{i}__{name}__.transform(...)``.
    """
    count = 0

    def transform_call(name: str) -> str:
        nonlocal count
        object_name = f"_patsy_stobj{count}__{name}__"
        if object_name not in transforms:
            return name
        count += 1
        return f"{object_name}.transform"

    return str(replace_bare_funcalls(code, transform_call))


def _check_expression(
    code: str, formula: str, covariate_names: Collection[str] | None
) -> None:
    """Refuse a Python expression of ``formula`` outside the grammar.

    Also refuse one naming a covariate missing from ``covariate_names``.
    """
    try:
        expression = ast.parse(code, mode="eval").body
    except SyntaxError:
        _refuse(formula, f"the invalid Python expression {code!r}")
    offending = _offending(expression)
    if offending is not None:
        _refuse(formula, offending)
    if covariate_names is None:
        return
    available = set(covariate_names)
    for name in sorted(_names(expression)):
        if name not in available:
            _refuse(
                formula,
                f"the name {name}, which is not a covariate (the covariates "
                f"are {sorted(map(str, available))})",
            )


def _offending(node: ast.AST) -> str | None:
    """Describe the first node of ``node`` outside the grammar, or return None."""
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
        if isinstance(node.op, ast.Pow):
            exponent = _number(node.right)
            if exponent is None or abs(exponent) > _MAX_EXPONENT:
                return (
                    f"the power {ast.unparse(node)}, whose exponent is not a "
                    f"number of magnitude at most {_MAX_EXPONENT}"
                )
            if any(
                isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Pow)
                for inner in ast.walk(node.left)
            ):
                return f"the nested power {ast.unparse(node)}"
        return _offending(node.left) or _offending(node.right)
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARY_OPERATORS):
            return f"the operator in {ast.unparse(node)}"
        return _offending(node.operand)
    if isinstance(node, ast.List | ast.Tuple):
        for element in node.elts:
            if _number(element) is None:
                return f"the list {ast.unparse(node)}, which holds more than numbers"
        return None
    if isinstance(node, ast.Call):
        return _offending_call(node)
    return f"the {type(node).__name__} expression {ast.unparse(node)}"


def _offending_call(call: ast.Call) -> str | None:
    """Describe what a call does outside the grammar, or return None."""
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
    # patsy calls C's contrast (its second argument).
    is_c = isinstance(function, ast.Name) and function.id == "C"
    if is_c and (
        len(call.args) > 1
        or any(keyword.arg == "contrast" for keyword in call.keywords)
    ):
        return f"the contrast in {ast.unparse(call)}"
    for node in _arguments(call):
        offending = _offending(node)
        if offending is not None:
            return offending
    return None


def _number(node: ast.expr) -> float | None:
    """Return the value of a (signed) int or float literal, or None otherwise."""
    sign = 1
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_OPERATORS):
        if isinstance(node.op, ast.USub):
            sign = -sign
        node = node.operand
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return sign * float(node.value)
    return None


def _names(node: ast.AST) -> set[str]:
    """Return the covariate names of a grammar-checked expression.

    These are its names, other than the functions it calls and ``np``.
    """
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return set()
    children: list[ast.AST] = (
        list(_arguments(node))
        if isinstance(node, ast.Call)
        else list(ast.iter_child_nodes(node))
    )
    return set().union(*(_names(child) for child in children))


def _arguments(call: ast.Call) -> list[ast.expr]:
    """Positional and keyword argument values of a call."""
    return [*call.args, *(keyword.value for keyword in call.keywords)]


def _numpy_member(node: ast.Attribute) -> str | None:
    """``name`` of ``np.<name>``, or None for any other attribute access."""
    if isinstance(node.value, ast.Name) and node.value.id == "np":
        return node.attr
    return None


def _refuse(formula: str, offending: str) -> NoReturn:
    """Raise the error that ``formula`` uses ``offending``."""
    msg = f"Transition formula {formula!r} uses {offending}, which is not allowed."
    raise ValidationError(msg, expected=_ALLOWED, got=offending, hint=_HINT)
