from app.services.context_chunking import split_text_semantically, summarize_text_chunks


def test_split_text_semantically_prefers_sentence_boundaries():
    text = (
        "First sentence explains the setup. "
        "Second sentence contains more detail. "
        "Third sentence keeps going for a bit longer."
    )

    chunks = split_text_semantically(text, 55)

    assert len(chunks) >= 2
    assert chunks[0].endswith(".")
    assert "Second sentence" in chunks[1]


def test_split_text_semantically_prefers_function_boundaries_for_code():
    text = """
def first_function():
    return "alpha"


def second_function():
    return "beta"
""".strip()

    chunks = split_text_semantically(text, 60)

    assert len(chunks) == 2
    assert "def first_function()" in chunks[0]
    assert "def second_function()" not in chunks[0]
    assert chunks[1].startswith("def second_function()")


def test_summarize_text_chunks_keeps_latest_tail_for_large_content():
    text = " ".join(f"Section {idx} has details." for idx in range(1, 8))

    chunks = summarize_text_chunks(text, per_chunk_chars=30, max_chunks=3)

    assert len(chunks) == 3
    assert chunks[-1].endswith("Section 7 has details.")
    assert "earlier chunks compacted" in chunks[1]


def test_summarize_text_chunks_preserves_fenced_code_shape():
    text = """
```python
def first():
    return "alpha"

def second():
    return "beta"
```
""".strip()

    chunks = summarize_text_chunks(text, per_chunk_chars=40, max_chunks=3)

    assert chunks
    assert chunks[0].startswith("[fenced python")
    assert "def first():" in chunks[0]


def test_summarize_text_chunks_preserves_diff_landmarks():
    text = """
diff --git a/app.py b/app.py
index 123..456 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,5 @@
-old_line = 1
+old_line = 2
+new_line = 3
""".strip()

    chunks = summarize_text_chunks(text, per_chunk_chars=70, max_chunks=3)

    assert chunks
    joined = " | ".join(chunks)
    assert "diff --git a/app.py b/app.py" in joined
    assert "@@ -1,3 +1,5 @@" in joined


def test_summarize_text_chunks_preserves_stack_trace_landmarks():
    text = """
Traceback (most recent call last):
  File "/app/main.py", line 10, in <module>
    run()
  File "/app/main.py", line 5, in run
    raise ValueError("boom")
ValueError: boom
""".strip()

    chunks = summarize_text_chunks(text, per_chunk_chars=70, max_chunks=3)

    assert chunks
    assert "Traceback (most recent call last):" in chunks[0]
    assert 'File "/app/main.py", line 10' in chunks[0]
    assert "ValueError: boom" in chunks[-1]
