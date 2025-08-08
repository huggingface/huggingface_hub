# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from unittest.mock import MagicMock

from huggingface_hub.inference._mcp.mcp_client import MCPClient


class TestMCPClient(unittest.TestCase):
    def setUp(self):
        self.client = MCPClient(model="test-model", provider="test-provider")

    def test_filter_tools_no_allowed_tools(self):
        """Test that _filter_tools returns all tools when no allowed_tools is specified."""
        # Create mock tools
        mock_tools = [
            MagicMock(name="tool1"),
            MagicMock(name="tool2"),
            MagicMock(name="tool3"),
        ]

        result = self.client._filter_tools(mock_tools, None)

        self.assertEqual(len(result), 3)
        self.assertEqual(result, mock_tools)

    def test_filter_tools_with_allowed_tools(self):
        """Test that _filter_tools correctly filters tools based on allowed_tools list."""
        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool3 = MagicMock()
        mock_tool3.name = "tool3"

        mock_tools = [mock_tool1, mock_tool2, mock_tool3]
        allowed_tools = ["tool1", "tool3"]

        result = self.client._filter_tools(mock_tools, allowed_tools)

        self.assertEqual(len(result), 2)
        self.assertIn(mock_tool1, result)
        self.assertIn(mock_tool3, result)
        self.assertNotIn(mock_tool2, result)

    def test_filter_tools_with_empty_allowed_tools(self):
        """Test that _filter_tools returns empty list when allowed_tools is empty."""
        mock_tools = [
            MagicMock(name="tool1"),
            MagicMock(name="tool2"),
        ]

        result = self.client._filter_tools(mock_tools, [])

        self.assertEqual(len(result), 0)

    def test_filter_tools_with_nonexistent_tools(self):
        """Test that _filter_tools handles non-existent tool names gracefully."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tools = [mock_tool1]

        # Include a non-existent tool in allowed_tools
        allowed_tools = ["tool1", "nonexistent_tool"]

        with self.assertLogs(level="WARNING") as log:
            result = self.client._filter_tools(mock_tools, allowed_tools)

        # Should only return existing tools
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_tool1)

        # Should log a warning about missing tools
        self.assertIn("not found on server", log.output[0])

    def test_filter_tools_all_nonexistent_tools(self):
        """Test that _filter_tools returns empty list when all allowed_tools are non-existent."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tools = [mock_tool1]

        allowed_tools = ["nonexistent_tool1", "nonexistent_tool2"]

        with self.assertLogs(level="WARNING") as log:
            result = self.client._filter_tools(mock_tools, allowed_tools)

        self.assertEqual(len(result), 0)
        self.assertIn("not found on server", log.output[0])


if __name__ == "__main__":
    unittest.main()
