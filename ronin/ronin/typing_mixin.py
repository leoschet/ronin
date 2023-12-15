from typing import Literal


ChatMessage = dict[Literal["role", "message"], str]
"""A chat message in a conversation with AI assistants."""
