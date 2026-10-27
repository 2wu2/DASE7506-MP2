"""Independent arithmetic examples, unsupported syntax, and rule semantics."""
import json
from pathlib import Path
import numpy as np
from evidence_features import checked_equations, safe_value, candidate_features, NAMES


def main():
    cases = [
        ('12 * 5 = 60', True), ('120 - 35 = 95', False),
        ('1 + 2 * 3 = 7', True), ('1 + 2 * 3 = 9', False),
        ('(1 + 2) * 3 = 9', True), ('1,200 / 3 = 400', True),
        (r'\frac{3}{4} \times 20 = 15', True),
        ('25% * 200 = 50', True), ('0.1 + 0.2 = 0.3', True),
        (r'\frac{7}{35} \times 100 = 20\%', True),
        ('1 / 4 = 25%', True),
        ('- $6 \u00d7 2 = $12', True), ('- 6 * 2 = 12', None),
        ('50% of the remaining 70 - 14 = 28', None),
        (r'40 + 88 + 66 = \boxed{194}', True),
        ('**12 * 5 = 60**', True), ('2**3 + 1 = 9', None),
        ('x + 2 = 5', None), ('x + 2 + 3 = 5', None),
        ('x^2 + 3 = 5', None), ('3 + 2 = 5x', None),
        ('3 + 2 = x', None), ('3 / 0 = 5', None),
        ('3 apples plus 2 apples = 5 apples', None),
        ('No equation, final answer 9.', None),
    ]
    for text, expected in cases:
        _, matches = checked_equations(text)
        if expected is None:
            assert not matches, (text, matches)
        else:
            assert len(matches) == 1 and matches[0]['exact'] == expected, (text, matches)
    rounded = checked_equations('10 / 3 = 3.33')[1]
    assert rounded[0]['rounded'] and not rounded[0]['wrong']
    # Unsupported code and resource-expensive operations never execute.
    for expression in ["__import__('os').system('whoami')", '2**1000000000', '1/0', 'a+2']:
        try:
            safe_value(expression)
        except (ValueError, SyntaxError, ZeroDivisionError):
            pass
        else:
            raise AssertionError(expression)
    # Deterministically generated, exact arithmetic is a useful independent oracle.
    rng = np.random.default_rng(7)
    for _ in range(100):
        a, b = map(int, rng.integers(1, 1000, size=2))
        for expression, expected in [(f'{a}+{b}', a+b), (f'{a}*{b}', a*b), (f'{a}-{b}', a-b)]:
            assert checked_equations(f'{expression} = {expected}')[1][0]['exact']
            assert checked_equations(f'{expression} = {expected+5}')[1][0]['wrong']
    v = dict(zip(NAMES, candidate_features('How many apples are left?', '12 - 5 = 7\n#### 8')))
    assert v['last_equation_differs_answer'] and not v['last_equation_matches_answer']
    v = dict(zip(NAMES, candidate_features('How many students are there?', '#### 2.5')))
    assert v['discrete_noninteger']
    v = dict(zip(NAMES, candidate_features('How many hours on average?', '#### 2.5')))
    assert not v['discrete_noninteger']
    v = dict(zip(NAMES, candidate_features('How many minutes?', '2 * 60 = 120\n#### 120 minutes')))
    assert not v['asked_minutes_but_final_hours']
    out = Path('runs/evidence_v4'); out.mkdir(parents=True, exist_ok=True)
    (out/'feature-checks.json').write_text(json.dumps(dict(explicit_cases=len(cases), generated_equations=600,
          fraction_and_percentage=True,unsupported_is_unknown=True,safe_ast=True,feature_dimensions=len(NAMES)), indent=2))
    print('Evidence feature checks passed:', len(NAMES), 'features')


if __name__ == '__main__':
    main()
