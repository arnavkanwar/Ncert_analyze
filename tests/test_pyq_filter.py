from src.pipeline.pyq_filter import filter_pyq_chunks, is_valid_question


def test_is_valid_question_rejects_noise() -> None:
    assert not is_valid_question("General Instructions:")
    assert not is_valid_question("Maximum Marks: 80   Time: 3 Hours")
    assert not is_valid_question("Page 2 of 8")
    assert not is_valid_question("Section B")


def test_is_valid_question_accepts_questions() -> None:
    assert is_valid_question("Q1. Which of the following is a redox reaction?")
    assert is_valid_question("State Henry's Law and give one application.")
    assert is_valid_question("Explain the role of enzymes in digestion.")


def test_filter_pyq_chunks_keeps_only_questions() -> None:
    raw_chunks = [
        {"text": "Section A: Multiple Choice Questions"},
        {"text": "Q1. Which of the following is a redox reaction?"},
        {"text": "General Instructions:"},
        {"text": "Q2. State Henry's Law and give one application."},
    ]

    clean_chunks = filter_pyq_chunks(raw_chunks)
    assert len(clean_chunks) == 2
    assert clean_chunks[0]["text"].startswith("Q1")
    assert clean_chunks[1]["text"].startswith("Q2")
