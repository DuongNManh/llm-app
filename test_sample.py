import pytest
from home_works.aie01.idiomatic import Summary
from home_works.weathercontroller import get_weather, Weather


def calculator(x: int, y: int, op: str):
    if op == "+":
        return x + y
    elif op == "-":
        return x - y
    elif op == "*":
        return x * y
    elif op == "/":
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y


def test_calculator():
    assert calculator(2, 3, "+") == 5
    assert calculator(2, 3, "-") == -1
    assert calculator(2, 3, "*") == 6
    assert calculator(6, 2, "/") == 3
    with pytest.raises(ValueError):
        calculator(6, 0, "/")


def test_summary():
    summary = Summary(word_count=10, text="This is a summary.")
    assert summary.word_count == 10
    assert summary.text == "This is a summary."


def test_call_llm():
    from home_works.aie01.idiomatic import call_llm
    import asyncio

    async def test():
        docs = ["This is a test document.", "Another document."]
        summaries = await call_llm(docs)
        assert len(summaries) == 2
        for summary in summaries:
            assert isinstance(summary, Summary)
            assert summary.word_count > 0
            assert summary.text.startswith("Summary:")

    asyncio.run(test())


def test_get_weather():
    import asyncio

    async def test():
        city = "Da Nang"
        weather = await get_weather(city)
        assert isinstance(weather, Weather)
        assert 15.0 <= weather.temperature <= 30.0
        assert 40.0 <= weather.humidity <= 80.0
        assert weather.description in ["Sunny", "Cloudy", "Rainy", "Windy"]

    asyncio.run(test())
