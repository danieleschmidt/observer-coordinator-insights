import pytest
from insights.profile import PersonProfile, QUADRANTS


def make_profile(**kwargs):
    defaults = {"name": "Test", "red": 50, "yellow": 30, "green": 40, "blue": 60}
    defaults.update(kwargs)
    return PersonProfile(**defaults)


def test_dominant_color():
    p = PersonProfile("Alice", red=80, yellow=20, green=30, blue=50)
    assert p.dominant_color == "red"


def test_dominant_color_blue():
    p = PersonProfile("Bob", red=10, yellow=20, green=30, blue=90)
    assert p.dominant_color == "blue"


def test_vector_length():
    p = make_profile()
    assert len(p.vector) == 4


def test_normalized_vector_sums_to_one():
    p = PersonProfile("C", red=25, yellow=25, green=25, blue=25)
    nv = p.normalized_vector
    assert abs(sum(nv) - 1.0) < 1e-9


def test_invalid_score_raises():
    with pytest.raises(ValueError):
        PersonProfile("D", red=150)


def test_traits_returns_list():
    p = PersonProfile("E", red=90)
    traits = p.traits()
    assert isinstance(traits, list)
    assert len(traits) > 0


def test_from_dict():
    d = {"name": "F", "red": 70, "yellow": 20, "green": 10, "blue": 40}
    p = PersonProfile.from_dict(d)
    assert p.name == "F"
    assert p.red == 70


def test_to_dict():
    p = make_profile(name="G")
    d = p.to_dict()
    assert d["name"] == "G"
    assert "red" in d
