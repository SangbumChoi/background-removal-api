from background_remove_sdk.agents import (
    TOOL_FUNCTIONS,
    execute_tool,
    get_anthropic_tools,
    get_openai_tools,
)


class TestSchemas:
    def test_openai_format(self):
        tools = get_openai_tools()
        assert {t["function"]["name"] for t in tools} == set(TOOL_FUNCTIONS)
        for tool in tools:
            assert tool["type"] == "function"
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            if tool["function"]["name"] != "list_models":
                assert "input_path" in params["properties"]
                assert "input_path" in params["required"]
                assert "model" in params["properties"]

    def test_anthropic_format(self):
        tools = get_anthropic_tools()
        assert {t["name"] for t in tools} == set(TOOL_FUNCTIONS)
        for tool in tools:
            assert tool["description"]
            assert tool["input_schema"]["type"] == "object"

    def test_schemas_are_copies(self):
        get_openai_tools()[0]["function"]["name"] = "mutated"
        assert get_openai_tools()[0]["function"]["name"] != "mutated"


class TestExecuteTool:
    def test_remove_background(self, patched_shared_remover, sample_image):
        result = execute_tool("remove_background", {"input_path": str(sample_image)})
        assert result["output_path"].endswith("sample_no_bg.png")
        assert result["width"] == 100 and result["height"] == 80

    def test_generate_mask_custom_output(self, patched_shared_remover, sample_image, tmp_path):
        out = tmp_path / "m.png"
        result = execute_tool(
            "generate_mask", {"input_path": str(sample_image), "output_path": str(out)}
        )
        assert result["output_path"] == str(out)
        assert out.exists()

    def test_extract_object(self, patched_shared_remover, sample_image):
        result = execute_tool(
            "extract_object_at_point", {"input_path": str(sample_image), "x": 30, "y": 20}
        )
        assert result["width"] == 40 and result["height"] == 30

    def test_unknown_tool(self):
        result = execute_tool("nonexistent", {})
        assert "Unknown tool" in result["error"]

    def test_bad_arguments_return_error(self):
        result = execute_tool("remove_background", {"wrong_arg": "x"})
        assert "error" in result

    def test_missing_file_returns_error(self, patched_shared_remover):
        result = execute_tool("remove_background", {"input_path": "/nope/missing.jpg"})
        assert "error" in result

    def test_point_on_background_returns_error(self, patched_shared_remover, sample_image):
        result = execute_tool(
            "extract_object_at_point", {"input_path": str(sample_image), "x": 90, "y": 70}
        )
        assert "No foreground object" in result["error"]
