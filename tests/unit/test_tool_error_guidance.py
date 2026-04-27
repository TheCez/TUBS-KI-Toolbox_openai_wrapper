from app.services.tool_error_guidance import guidance_for_tool_errors, guidance_for_tool_successes


def test_guidance_for_tool_errors_extracts_file_and_symbol_metadata():
    hints = guidance_for_tool_errors(
        [
            {
                "is_error": True,
                "text": (
                    "Update(C:\\personal\\portfolio\\cowork_test_project\\src\\components\\PrimaryButton.tsx)\n"
                    "Error: String to replace not found in file.\n"
                    "String: const baseClassName =\n"
                    "  'inline-flex items-center justify-center'"
                ),
            }
        ]
    )

    joined = "\n".join(hints)
    assert "Wrapper repair hint:" in joined
    assert "target file: `C:\\personal\\portfolio\\cowork_test_project\\src\\components\\PrimaryButton.tsx`" in joined
    assert "likely stable anchor: `const baseClassName`" in joined


def test_guidance_for_tool_errors_extracts_component_anchor_from_jsx():
    hints = guidance_for_tool_errors(
        [
            {
                "is_error": True,
                "text": (
                    "Update(C:\\personal\\portfolio\\cowork_test_project\\src\\components\\SectionCard.tsx)\n"
                    "Error: String to replace not found in file.\n"
                    "String:     <SectionCard className=\"test\">"
                ),
            }
        ]
    )

    joined = "\n".join(hints)
    assert "SectionCard.tsx" in joined
    assert "likely stable anchor: `SectionCard`" in joined


def test_guidance_for_tool_successes_adds_task_completion_hint_for_file_write():
    hints = guidance_for_tool_successes(
        [
            {
                "is_error": False,
                "text": "Wrote 12 lines to fibonacci.py",
            }
        ]
    )

    joined = "\n".join(hints)
    assert "Wrapper completion hint:" in joined
    assert "mark the related task or todo as completed" in joined
    assert "successful file operation on `fibonacci.py`" in joined
