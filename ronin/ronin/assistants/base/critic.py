import attrs

from ronin.prompts.templates import (
    UserMessageTemplate,
)
from ronin.typing_mixin import ChatMessage

from .chat import ChatAssistant
from .registry import AssistantRegister


@AssistantRegister.register("base-critic-assistant")
@attrs.define
class BaseCriticAssistant(ChatAssistant):
    suggestion_feedback_prompt: UserMessageTemplate = attrs.field(
        factory=UserMessageTemplate.with_dummy_template, repr=False
    )

    def build_suggestion_feedback_message(self, message: str, **kwargs) -> ChatMessage:
        """Build the user's chat message."""
        params = kwargs
        params[self.suggestion_feedback_prompt.message_prompt_parameter] = message
        return self.suggestion_feedback_prompt.fill(**params)

    def chat(
        self,
        message: str,
        user_message_kwargs: dict | None = None,
        system_message_kwargs: dict | None = None,
    ) -> ChatMessage:
        """Chat with a critic."""
        if not user_message_kwargs:
            user_message_kwargs = {}

        suggestions_feedback = user_message_kwargs.pop("suggestions_feedback", "")

        if suggestions_feedback:
            suggestions_feedback_message = self.build_suggestion_feedback_message(
                message=suggestions_feedback, **user_message_kwargs
            )

            self.history.append(suggestions_feedback_message)

        return super().chat(message, user_message_kwargs, system_message_kwargs)
