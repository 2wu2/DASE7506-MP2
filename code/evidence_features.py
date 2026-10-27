"""Conservative mathematical evidence from supplied question/candidate text.

No reference answers, external models, execution of candidate code, or ID use.
Unparsed expressions are unknown, not arithmetic errors. Most constraints
are deliberately soft features, not vetoes on candidate selection.
"""
import ast
from fractions import Fraction
import math
import re
import numpy as np
from common import parse_answer

ARITHMETIC = [
    'equations_seen_log', 'equations_checked_log', 'check_coverage',
    'equations_exact_log', 'equations_rounded_log', 'equations_wrong_log',
    'wrong_fraction', 'any_wrong', 'mean_relative_error', 'max_relative_error',
    'last_checked_wrong', 'checked_plus_log', 'checked_minus_log',
    'checked_multiply_log', 'checked_divide_log',
]
CONSISTENCY = [
    'answer_parsed', 'has_checked_equation', 'last_equation_matches_answer',
    'last_equation_differs_answer', 'any_equation_matches_answer',
    'last_computation_matches_answer', 'multiple_final_markers',
    'final_markers_disagree', 'answer_is_integer', 'answer_negative',
    'answer_zero', 'answer_magnitude_log',
]
CONSTRAINTS = [
    'question_discrete_count', 'discrete_noninteger', 'discrete_negative',
    'question_remainder', 'remainder_without_subtraction_cue',
    'question_average', 'average_without_division_cue',
    'question_total', 'total_without_add_or_multiply_cue',
    'question_percentage', 'percentage_without_scaling_cue',
    'question_time_conversion', 'time_conversion_without_60',
    'asked_minutes_but_final_hours', 'asked_hours_but_final_minutes',
    'question_money', 'question_number_overlap', 'novel_numbers_fraction',
    'explicit_percent_answer_outside_range',
]
NAMES = ARITHMETIC + CONSISTENCY + CONSTRAINTS
BLOCKS = {'arithmetic': slice(0, len(ARITHMETIC)),
          'consistency': slice(len(ARITHMETIC), len(ARITHMETIC)+len(CONSISTENCY)),
          'constraints': slice(len(ARITHMETIC)+len(CONSISTENCY), len(NAMES))}
NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
NUMBER_RE = re.compile(NUM)


def normalize_math(text):
    t = text.replace('\u2212', '-').replace('\u00d7', '*').replace('\u00f7', '/')
    t = re.sub(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b', lambda m: m[0].replace(',', ''), t)
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/')
    t = t.replace('\\left', '').replace('\\right', '')
    t = t.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    for _ in range(6):
        newer = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', t)
        if newer == t:
            break
        t = newer
    t = re.sub(r'\\boxed\{([^{}]+)\}', r'(\1)', t)
    t = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', ' ', t)
    t = t.replace('\\%', '%')
    t = re.sub(r'(' + NUM + r')\s*%', r'(\1/100)', t)
    # Formatting is removed, but math unknowns/functions remain to prevent
    # accidentally interpreting a suffix of an algebraic expression.
    t = re.sub(r'(?<!\w)\*\*(?=\S)|(?<=\S)\*\*(?!\w)', '', t)
    t = t.replace('$', '').replace('\\,', ' ').replace('\\;', ' ')
    t = t.replace('\\(', ' ').replace('\\)', ' ').replace('\\[', ' ').replace('\\]', ' ')
    return t


def safe_value(expression):
    """Bounded arithmetic AST evaluator; never eval/exec candidate material."""
    s = expression.strip()
    if not s or len(s) > 180 or re.fullmatch(r'[0-9.()+*/\-\s]+', s) is None:
        raise ValueError('unsupported expression')
    tree = ast.parse(s, mode='eval')
    if len(list(ast.walk(tree))) > 75:
        raise ValueError('expression too complex')
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(s, node)
            if len(token) > 35:
                raise ValueError('number too long')
            value = Fraction(token)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = a+b
            elif isinstance(node.op, ast.Sub): value = a-b
            elif isinstance(node.op, ast.Mult): value = a*b
            else: value = a/b
        else:
            raise ValueError('unsupported node')
        if value.numerator.bit_length() > 180 or value.denominator.bit_length() > 180:
            raise ValueError('arithmetic magnitude too large')
        return value
    return visit(tree)


def checked_equations(text):
    seen = 0; checked = []
    for line in text.splitlines():
        # A Markdown bullet is not a negative operand. For unformatted '- 6'
        # the intent is ambiguous, so count it as seen but leave it unchecked.
        if re.match(r'^\s*-\s+\d', line):
            seen += line.count('=')
            continue
        line = re.sub(r'^\s*[-*]\s+(?=\$|\*\*|\\\()', '', line)
        parts = line.split('=')
        seen += len(parts)-1
        for original_left, original_right in zip(parts, parts[1:]):
            left = normalize_math(original_left)
            percent = re.match(r'\s*(' + NUM + r')\s*\\?%',
                               original_right.replace('$', '').replace('**', ''))
            # Written percentage calculations commonly use 0.1*100=10%:
            # the RHS denotes percentage points in this convention. Preserve
            # the numeric display instead of incorrectly calling it 10=0.1.
            percent_points = bool(percent and re.search(r'\*\s*100\b', left))
            if percent_points:
                right = normalize_math(re.sub(r'(' + NUM + r')\s*\\?%', r'\1', original_right, count=1))
            else:
                right = normalize_math(original_right)
            lm = re.search(r'[0-9.()+*/\-\s]+$', left)
            rm = re.match(r'[0-9.()+*/\-\s]+', right)
            if lm is None or rm is None:
                continue
            lhs, rhs = lm[0].strip(), rm[0].strip()
            # A suffix after a variable, exponent or unknown math operator is
            # not a complete arithmetic expression. Prose separated by space
            # may precede a standalone expression (e.g. 'Total: 2 + 3').
            prefix = left[:lm.start()].rstrip()
            if re.search(r'\b(?:of|times|plus|minus|divided by|multiplied by)\s*(?:the\s+)?(?:remaining\s*)?$', prefix):
                continue
            if prefix and prefix[-1] in '^_<>!%':
                continue
            if prefix and prefix[-1].isalpha() and not lm[0].startswith(' '):
                continue
            if prefix and prefix[-1].isalpha() and (lhs.startswith(('+', '-')) or
                    re.search(r'(?:^|\W)[A-Za-z]$', prefix)):
                continue
            suffix = right[rm.end():]
            if suffix and (suffix[0] in '^_<>!%(' or suffix[0].isdigit()):
                continue
            # Variable products like 3x or 5a are not a numeric RHS.
            if suffix and suffix[0].isalpha() and not rm[0].endswith(' '):
                continue
            if not re.search(r'[+*/]|\d\s*-', lhs):
                continue
            try:
                a, b = safe_value(lhs), safe_value(rhs)
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
                continue
            residual = float(abs(a-b)/max(Fraction(1), abs(a), abs(b)))
            absolute = float(abs(a-b))
            exact = a == b
            # Rounded is a distinct feature, never called exactly correct.
            decimal = re.fullmatch(r'[-+]?\d*\.(\d+)', rhs)
            tolerance = .5 * 10**(-len(decimal[1])) if decimal else 0.
            rounded = not exact and tolerance > 0 and absolute <= tolerance+1e-12
            checked.append(dict(lhs=lhs, rhs=rhs, expected=str(a), stated=str(b),
                                answer_stated=str(Fraction(percent[1])) if percent else str(b),
                                answer_expected=str(a*100) if percent and not percent_points else str(a),
                                exact=exact, rounded=rounded,
                                wrong=not (exact or rounded), residual=residual))
    return seen, checked


def numeric_values(text):
    values = set()
    for m in NUMBER_RE.finditer(normalize_math(text)):
        try:
            value = Fraction(m[0])
            if value.numerator.bit_length() < 100:
                values.add(value)
        except (ValueError, ZeroDivisionError):
            pass
    return values


def candidate_features(question, text):
    seen, checked = checked_equations(text)
    parsed = parse_answer(text)
    answer = Fraction(parsed) if parsed is not None else None
    count = len(checked); wrong = sum(c['wrong'] for c in checked)
    residuals = [min(c['residual'], 2.) for c in checked]
    arithmetic = [np.log1p(seen), np.log1p(count), count/max(1, seen),
                  np.log1p(sum(c['exact'] for c in checked)),
                  np.log1p(sum(c['rounded'] for c in checked)), np.log1p(wrong),
                  wrong/max(1, count), wrong > 0,
                  sum(residuals)/max(1, count), max(residuals, default=0.),
                  checked[-1]['wrong'] if checked else False]
    for op in ('+', '-', '*', '/'):
        arithmetic.append(np.log1p(sum(op in c['lhs'] for c in checked)))
    last_match = bool(answer is not None and checked and Fraction(checked[-1]['answer_stated']) == answer)
    computed_match = bool(answer is not None and checked and Fraction(checked[-1]['answer_expected']) == answer)
    markers = re.findall(r'\\boxed\{([^{}]+)\}|####([^\n]*)', text)
    finals = [parse_answer('#### '+(a or b)) for a, b in markers]
    finals = [v for v in finals if v is not None]
    magnitude = min(50., math.log1p(abs(float(answer)))) if answer is not None else 0.
    consistency = [answer is not None, count > 0, last_match,
                   bool(answer is not None and checked and not last_match),
                   bool(answer is not None and any(Fraction(c['answer_stated']) == answer for c in checked)),
                   computed_match, len(finals) > 1, len(set(finals)) > 1,
                   answer.denominator == 1 if answer is not None else False,
                   answer < 0 if answer is not None else False,
                   answer == 0 if answer is not None else False, magnitude]
    q = question.lower(); t = text.lower(); mathtext = normalize_math(text)
    # Only simple requested discrete counts, not average rates or measurements.
    discrete = bool(re.search(r'how many (?:people|persons|children|students|boys|girls|'
                              r'apples|oranges|eggs|books|tickets|cars|dogs|cats|coins|'
                              r'chairs|tables|pencils|pens|marbles|balls|bottles)\b', q))
    discrete = discrete and not any(w in q for w in ('average', 'per ', 'each hour'))
    remainder = bool(re.search(r'\b(?:left|remaining|remain|remainder)\b', q))
    average = 'average' in q
    total = bool(re.search(r'\b(?:total|altogether|in all)\b', q))
    percentage = '%' in q or 'percent' in q
    conversion = bool(re.search(r'\bhours?\b', q) and re.search(r'\bminutes?\b', q))
    ask_minutes = bool(re.search(r'how many minutes|how much time.*minutes', q))
    ask_hours = bool(re.search(r'how many hours|how much time.*hours', q))
    tail = t.splitlines()[-1] if t.splitlines() else ''
    qnums, tnums = numeric_values(question), numeric_values(text)
    if answer is not None:
        tnums.discard(answer)
    constraints = [discrete,
                   discrete and answer is not None and answer.denominator != 1,
                   discrete and answer is not None and answer < 0,
                   remainder, remainder and not ('-' in mathtext or re.search(r'subtract|minus|difference|deduct', t)),
                   average, average and not ('/' in mathtext or re.search(r'divid|per ', t)),
                   total, total and not ('+' in mathtext or '*' in mathtext or re.search(r'add|sum|multipl', t)),
                   percentage, percentage and not re.search(r'100|0\.\d|percent|%', t),
                   conversion, conversion and not re.search(r'\b60\b', t),
                   ask_minutes and bool(re.search(r'\bhours?\b', tail)) and 'minute' not in tail,
                   ask_hours and bool(re.search(r'\bminutes?\b', tail)) and 'hour' not in tail,
                   bool(re.search(r'\$|dollar|cent|cost|price', q)),
                   len(qnums & tnums)/max(1, len(qnums)),
                   len(tnums-qnums)/max(1, len(tnums)),
                   bool(re.search(r'what (?:percentage|percent) of', q)) and answer is not None and not 0 <= answer <= 100]
    values = np.array(arithmetic+consistency+constraints, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) != len(NAMES):
        raise ValueError('invalid evidence feature vector')
    return values


def evidence(bank):
    if not bank:
        return np.empty((0, 4, len(NAMES)))
    return np.array([[candidate_features(q['question'], c['text']) for c in q['candidates']]
                     for q in bank], dtype=np.float64)


def select_block(values, name):
    return values if name == 'all' else values[..., BLOCKS[name]]
