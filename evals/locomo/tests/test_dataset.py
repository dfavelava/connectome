from locomo_eval.dataset import normalize_evidence, parse_sample, parse_session_date

RAW_SAMPLE = {
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Look at this.", "blip_caption": "a photo of a sunset"},
        ],
        "session_2_date_time": "not a date",
        "session_2": [{"speaker": "Caroline", "dia_id": "D2:1", "text": "Back again."}],
        # Date-only session entries with no turns exist in locomo10.json.
        "session_3_date_time": "9:00 am on 1 June, 2023",
    },
    "qa": [
        {"question": "When?", "answer": "7 May 2023", "evidence": ["D1:1"], "category": 2},
        {"question": "Adversarial?", "adversarial_answer": "x", "evidence": ["D1:2; D2:1"], "category": 5},
        {"question": "No evidence?", "answer": "y", "evidence": [], "category": 4},
    ],
}


def test_parse_session_date():
    assert parse_session_date("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00+00:00"
    assert parse_session_date("12:05 am on 1 January, 2024") == "2024-01-01T00:05:00+00:00"
    assert parse_session_date("garbage") is None


def test_normalize_evidence_handles_malformed_entries():
    assert normalize_evidence(["D8:6; D9:17"]) == ("D8:6", "D9:17")
    assert normalize_evidence(["D9:1 D4:4 D4:6", "D4:4"]) == ("D9:1", "D4:4", "D4:6")
    assert normalize_evidence(["D:11:26"]) == ("D11:26",)
    assert normalize_evidence(["D"]) == ()
    assert normalize_evidence([]) == ()


def test_parse_sample():
    sample = parse_sample(RAW_SAMPLE)
    assert sample.sample_id == "conv-1"
    assert [t.dia_id for t in sample.turns] == ["D1:1", "D1:2", "D2:1"]
    assert sample.turns[0].occurred_at == "2023-05-08T13:56:00+00:00"
    assert sample.turns[0].session_date == "1:56 pm on 8 May, 2023"
    assert sample.turns[1].text == "Look at this. [shares an image: a photo of a sunset]"
    assert sample.turns[2].occurred_at is None

    assert [q.category_name for q in sample.qa] == ["temporal", "adversarial", "single-hop"]
    assert sample.qa[1].evidence == ("D1:2", "D2:1")
    assert sample.qa[2].evidence == ()
