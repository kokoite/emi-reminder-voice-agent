from datetime import date

import pytest

from spoken import date_in_words, digits, due_status, number_in_words, rupees


@pytest.mark.parametrize(
    ("amount", "spoken"),
    [
        (12500, "twelve thousand five hundred rupees"),
        (4999, "four thousand nine hundred ninety nine rupees"),
        (124000, "one lakh twenty four thousand rupees"),
        (100000, "one lakh rupees"),
        (2500000, "twenty five lakh rupees"),
        (10_050_000, "one crore fifty thousand rupees"),
        (1, "one rupee"),
    ],
)
def test_rupees_use_the_indian_system(amount, spoken):
    assert rupees(amount) == spoken


def test_small_numbers():
    assert number_in_words(0) == "zero"
    assert number_in_words(15) == "fifteen"
    assert number_in_words(40) == "forty"
    assert number_in_words(101) == "one hundred one"


@pytest.mark.parametrize(
    ("day", "spoken"),
    [
        (date(2026, 12, 5), "the fifth of December"),
        (date(2026, 1, 1), "the first of January"),
        (date(2026, 3, 22), "the twenty second of March"),
        (date(2026, 7, 30), "the thirtieth of July"),
        (date(2026, 8, 31), "the thirty first of August"),
    ],
)
def test_dates(day, spoken):
    assert date_in_words(day) == spoken


@pytest.mark.parametrize(
    ("days_overdue", "spoken"),
    [
        (5, "five days overdue"),
        (1, "one day overdue"),
        (0, "due today"),
        (-1, "due tomorrow"),
        (-3, "due in three days"),
    ],
)
def test_due_status(days_overdue, spoken):
    assert due_status(days_overdue) == spoken


def test_digits_are_read_one_by_one():
    assert digits("4821") == "four eight two one"
