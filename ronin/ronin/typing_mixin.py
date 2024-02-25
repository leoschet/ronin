from typing import Literal


ChatMessage = dict[Literal["role", "content"], str]
"""A chat message in a conversation with AI assistants."""
