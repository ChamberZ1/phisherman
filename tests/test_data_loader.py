import pandas as pd
import io

from src import data_loader


def _make_csv(content: str) -> str:
    return io.StringIO(content)


def test_normalize_label():
    mapping = {
        "Phishing": 1,
        "spam": 1,
        "SAFE": 0,
        "ham": 0,
        "1": 1,
        "0": 0,
        True: 1,
        False: 0,
        None: None,
        "unknown": None,
    }
    for inp, expected in mapping.items():
        assert data_loader.normalize_label(inp) == expected
    # floats should also be handled
    assert data_loader.normalize_label(0.0) == 0
    assert data_loader.normalize_label(1.0) == 1
    # numpy numeric types if available
    try:
        import numpy as np
        assert data_loader.normalize_label(np.float64(0)) == 0
        assert data_loader.normalize_label(np.float64(1)) == 1
    except ImportError:
        pass


def test_load_7col_with_label():
    csv = _make_csv("""sender,receiver,date,subject,body,label, urls
user@a.com,b@b.com,2020-01-01,hello,test,Phishing, 1
""")
    df = data_loader.load_7col(csv, source="foo")
    assert df.iloc[0]["label"] == 1
    assert "text_combined" in df.columns

def test_load_7col_safe():
    csv = _make_csv("""sender,receiver,date,subject,body,label, urls
user@a.com,b@b.com,2020-01-01,hello,test,Safe, 1
""")
    df = data_loader.load_7col(csv, source="foo")
    assert df.iloc[0]["label"] == 0
    assert "text_combined" in df.columns



def test_load_email_text_type():
    csv = _make_csv("""Email Text,Email Type
body text,Safe
""")
    df = data_loader.load_email_text_type(csv, source="bar")
    assert df.iloc[0]["label"] == 0


def test_load_email_text_type_verbose():
    csv = _make_csv("""Email Text,Email Type
foo,Safe Email
bar,Phishing Email
""")
    df = data_loader.load_email_text_type(csv, source="baz")
    assert df.iloc[0]["label"] == 0
    assert df.iloc[1]["label"] == 1


def test_subject_body_label():
    csv = _make_csv("""subject,body,label
s, b, spam
""")
    df = data_loader.load_subject_body_label(csv, source="baz")
    assert "text_combined" in df.columns
    assert "from_address" in df.columns 
    assert "source" in df.columns
    assert "subject" in df.columns
    assert "body" in df.columns
    assert "label" in df.columns
    assert "has_url" in df.columns
    assert df.iloc[0]["label"] == 1


def test_text_combined():
    csv = _make_csv("""text_combined,label
the whole message,0
""")
    df = data_loader.load_text_combined(csv, source="qux")
    assert df.iloc[0]["body"] == "the whole message"
    assert df.iloc[0]["label"] == 0


def test_load_all():
    configs = [
        {"load_fn": data_loader.load_email_text_type, "path": _make_csv("""Email Text,Email Type
foo,Phishing
"""), "source": "a"},
        {"load_fn": data_loader.load_text_combined, "path": _make_csv("""text_combined,label
bar,1
"""), "source": "b"},
    ]
    # make a copy to compare later
    orig = [dict(c) for c in configs]
    df = data_loader.load_all(configs)
    assert len(df) == 2
    assert df["source"].tolist() == ["a", "b"]
    # calling again should still work and configs should be intact
    df2 = data_loader.load_all(configs)
    assert len(df2) == 2
    assert configs == orig

def tests_load_all_loader_not_specified():
    configs = [
        {"path": _make_csv("""Email Text,Email Type
foo,Phishing
"""), "source": "a"},
        {"path": _make_csv("""text_combined,label
bar,1
"""), "source": "b"},
    ]
    # make a copy to compare later
    orig = [dict(c) for c in configs]
    df = data_loader.load_all(configs)
    assert len(df) == 2
    assert df["source"].tolist() == ["a", "b"]
    # calling again should still work and configs should be intact
    df2 = data_loader.load_all(configs)
    assert len(df2) == 2
    assert configs == orig

def test_load_all_keep_extra():
    configs = [
        {"load_fn": data_loader.load_email_text_type, "path": _make_csv("""Email Text,Email Type
foo,Safe
"""), "source": "x", "keep_extra": True},
    ]
    df = data_loader.load_all(configs)
    # loader should respect config-specified keep_extra (True) without TypeError
    assert "Email Text" in df.columns

def test_load_all_keep_extra_false():
    configs = [
        {"load_fn": data_loader.load_email_text_type, "path": _make_csv("""Email Text,Email Type
foo,Safe
"""), "source": "x", "keep_extra": False},
    ]
    df = data_loader.load_all(configs)
    # loader should respect config-specified keep_extra (False) without TypeError
    assert "Email Text" not in df.columns
    assert df.iloc[0]["has_url"] is None   
    assert "text_combined" in df.columns
    assert "from_address" in df.columns 
    assert "source" in df.columns
    assert "subject" in df.columns
    assert "body" in df.columns
    assert "label" in df.columns
    assert "has_url" in df.columns


def test_load_all_load_7col_with_label():
    configs = [
        {"path": _make_csv("""sender,receiver,date,subject,body,label,urls
user@a.com,b@b.com,2020-01-01,hello,test,Safe,1
"""), "source": "x"},
    ]
    df = data_loader.load_all(configs)

    assert "text_combined" in df.columns
    assert "from_address" in df.columns 
    assert "source" in df.columns
    assert "subject" in df.columns
    assert "body" in df.columns
    assert "label" in df.columns
    assert "has_url" in df.columns
    assert "sender" not in df.columns 
    assert df.iloc[0]["label"] == 0
    assert df.iloc[0]["has_url"] == 1
