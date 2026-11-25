"""Numbers and dates the way a person says them on a call (English, Indian system).

12500 -> "twelve thousand five hundred rupees"; 124000 -> "one lakh twenty four thousand
rupees"; 2026-12-05 -> "the fifth of December". The LLM gets these ready to speak, so it
never formats numbers itself.
"""

from datetime import date

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
    8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
    14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 30: "thirtieth",
}  # fmt: skip
_MONTHS = (
    "January February March April May June July August September October November December"
).split()


def _below_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[ones]}"


def _below_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = [f"{_ONES[hundreds]} hundred"] if hundreds else []
    if rest:
        parts.append(_below_hundred(rest))
    return " ".join(parts)


def number_in_words(n: int) -> str:
    """Whole number in words, Indian grouping: crore, lakh, thousand, hundred."""
    if n < 0:
        raise ValueError("negative amounts are not spoken")
    if n == 0:
        return "zero"
    parts = []
    for unit, size in (("crore", 10_000_000), ("lakh", 100_000), ("thousand", 1_000)):
        count, n = divmod(n, size)
        if count:
            parts.append(f"{number_in_words(count)} {unit}")
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def rupees(amount: int) -> str:
    return f"{number_in_words(amount)} rupee" + ("" if amount == 1 else "s")


def ordinal(n: int) -> str:
    if n in _ORDINALS:
        return _ORDINALS[n]
    tens, ones = divmod(n, 10)
    return f"{_TENS[tens]} {_ORDINALS[ones]}"


def date_in_words(d: date) -> str:
    return f"the {ordinal(d.day)} of {_MONTHS[d.month - 1]}"


def due_status(days_overdue: int) -> str:
    """Positive: overdue by that many days. Zero: due today. Negative: due in that many days."""
    if days_overdue == 0:
        return "due today"
    if days_overdue == 1:
        return "one day overdue"
    if days_overdue > 1:
        return f"{number_in_words(days_overdue)} days overdue"
    if days_overdue == -1:
        return "due tomorrow"
    return f"due in {number_in_words(-days_overdue)} days"


def digits(text: str) -> str:
    """Read digits one by one: '4821' -> 'four eight two one'."""
    return " ".join(_ONES[int(c)] for c in text)
