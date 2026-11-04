"""The grammar a covariate-transition formula read from a saved model must use.

patsy evaluates every term of a formula as Python, so a formula such as
``"__import__('os').system(...)"`` would run code when a design matrix is
built. A formula the user writes in their own session is their own code and is
not checked; one read from a saved model file may be someone else's, so
``load_model`` checks it against a small grammar before patsy evaluates it
(unless told the file is trusted).

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
- a call of one of the documented transforms (``_FUNCTIONS``) or elementwise
  numpy functions (``_NUMPY_FUNCTIONS``), with arguments, and keyword
  arguments, that are themselves allowed;
- ``C(data, contrast, levels)``, whose contrast (which patsy calls) must be a
  call of one of patsy's contrasts (``_CONTRASTS``) with literal arguments,
  e.g. ``C(arm, Treatment('b'))``, and whose levels are a list or tuple of
  literals, e.g. ``levels=['a', 'b']``. These are the only places a string
  literal may appear.

Everything else is refused: other attribute access, subscripts, strings
elsewhere, lambdas, comprehensions, conditionals, comparisons, bitwise
operators, line breaks and backslashes, and calls of anything else. Validating
the factor code patsy itself extracted means the check sees exactly the Python
that patsy would evaluate, so a string cannot be split one way for the check
and another way for evaluation.

patsy looks up a name it does not find in the covariates in the caller's
namespace and Python's builtins (``breakpoint``, ``exit``, ...). Wherever the
covariates are known, every covariate name of the formula must therefore be
one of them (``covariate_names``).
"""

import ast
from collections.abc import Collection
from typing import NoReturn

from patsy import ModelDesc, PatsyError

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
# Called as ``np.<name>(...)``: elementwise functions of their arguments.
_NUMPY_FUNCTIONS = frozenset(
    {
        "abs",
        "arccos",
        "arcsin",
        "arctan",
        "arctan2",
        "ceil",
        "cos",
        "exp",
        "expm1",
        "floor",
        "log",
        "log10",
        "log1p",
        "log2",
        "maximum",
        "minimum",
        "sign",
        "sin",
        "sqrt",
        "square",
        "tan",
    }
)
# patsy's contrasts, called with literal arguments as the contrast of C().
_CONTRASTS = frozenset({"Diff", "Helmert", "Poly", "Sum", "Treatment"})
# The parameters of patsy's C, in order.
_C_PARAMETERS = ("data", "contrast", "levels")
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
    + " (C takes its data, levels= as a list of literals, and as contrast a "
    "call of "
    + ", ".join(sorted(_CONTRASTS))
    + " with literal arguments, e.g. C(arm, Treatment('b')); string literals "
    "are allowed only there)"
)
_HINT = (
    "patsy evaluates a formula as Python, so a formula read from a saved "
    "model file is limited to this grammar. If you trust the file, load it "
    "with load_model(path, trust_formulas=True)."
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


def _check_expression(
    code: str, formula: str, covariate_names: Collection[str] | None
) -> None:
    """Refuse a Python expression of ``formula`` outside the grammar, or one
    naming a covariate missing from ``covariate_names``."""
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
    if isinstance(function, ast.Name) and function.id == "C":
        return _offending_c(call)
    for node in _arguments(call):
        offending = _offending(node)
        if offending is not None:
            return offending
    return None


def _offending_c(call: ast.Call) -> str | None:
    """Description of what a call of patsy's ``C`` does outside the grammar,
    or None.

    patsy calls the contrast, so it must be a call of one of ``_CONTRASTS``
    with literal arguments; the levels must be a list or tuple of literals.
    """
    if len(call.args) > len(_C_PARAMETERS):
        return f"the arguments of {ast.unparse(call)}"
    arguments = dict(zip(_C_PARAMETERS, call.args, strict=False))
    for keyword in call.keywords:
        if keyword.arg not in _C_PARAMETERS or keyword.arg in arguments:
            return f"the argument {keyword.arg}= in {ast.unparse(call)}"
        arguments[keyword.arg] = keyword.value
    contrast = arguments.get("contrast")
    if contrast is not None and not _is_contrast(contrast):
        return (
            f"the contrast {ast.unparse(contrast)} in {ast.unparse(call)}, which "
            "is not a call of " + ", ".join(sorted(_CONTRASTS)) + " with literal "
            "arguments"
        )
    levels = arguments.get("levels")
    if levels is not None and not (
        isinstance(levels, ast.List | ast.Tuple)
        and all(_is_scalar_literal(element) for element in levels.elts)
    ):
        return (
            f"the levels {ast.unparse(levels)} in {ast.unparse(call)}, which "
            "are not a list of literals"
        )
    data = arguments.get("data")
    return None if data is None else _offending(data)


def _is_contrast(node: ast.expr) -> bool:
    """Whether ``node`` is a call of one of patsy's contrasts whose arguments
    are literals (or lists or tuples of literals)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CONTRASTS
        and all(keyword.arg is not None for keyword in node.keywords)
        and all(_is_literal(argument) for argument in _arguments(node))
    )


def _is_literal(node: ast.expr) -> bool:
    """Whether ``node`` is a scalar literal or a list or tuple of them."""
    if isinstance(node, ast.List | ast.Tuple):
        return all(_is_scalar_literal(element) for element in node.elts)
    return _is_scalar_literal(node)


def _is_scalar_literal(node: ast.expr) -> bool:
    """Whether ``node`` is a (signed) number, a string, True, False or None."""
    if _number(node) is not None:
        return True
    return isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, bool | str)
    )


def _number(node: ast.expr) -> float | None:
    """The value of a (signed) int or float literal, or None for anything else."""
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
    """Covariate names of a grammar-checked expression: its names, other than
    the functions it calls and ``np``."""
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
