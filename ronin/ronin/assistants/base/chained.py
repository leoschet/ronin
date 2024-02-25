import attrs

from ronin.prompts.templates import (
    SystemPromptTemplate,
)
from ronin.typing_mixin import ChatMessage

from .chat import ChatAssistant, ProactiveChatAssistant
from .registry import AssistantRegister, AssistantRegisteredDataType


@AssistantRegister.register("base-chained-assistants-team")
@attrs.define(kw_only=True)
class ChainedAssistantsTeam:
    """A team of chained assistants.

    Attributes
    ----------
    assistants : list[ChatAssistant]
        The team of assistants.
    """

    assistants: list[ChatAssistant]

    def chat(
        self, message: str, **kwargs
    ) -> tuple[ChatMessage, list[list[ChatMessage]]]:
        """Chat with the team of assistants.

        The output from one assistant is passed as input to the next assistant.

        Returns
        -------
        tuple of ChatMessage and list of list of ChatMessage
            The response from the last assistant and the history of each assistant.
        """
        history = []
        response = message
        for assistant in self.assistants:
            response = assistant.chat(response, **kwargs)
            history.append(assistant.history)

        return response, history


@AssistantRegister.register(
    "base-chained-assistants-team",
    register_as=AssistantRegisteredDataType.BUILDER,
)
@attrs.define(kw_only=True)
class ChainedAssistantsTeamBuilder(ProactiveChatAssistant):
    """Builder for ChainedAssistantsTeam."""

    # Prefix with _ to avoid name clashes with the parent class.
    _system_prompt: str
    _steps_prompts: list[str]

    def build(self) -> ChainedAssistantsTeam:
        ChainedAssistantsTeam(
            assistants=[
                ChatAssistant(
                    chat_node=self.chat_node,
                    chat_system_prompt=SystemPromptTemplate.from_str(
                        self._system_prompt
                    ),
                    # TODO
                    # chat_user_prompt: ...,
                    # chat_assistant_prompt: ...,
                    # chat_system_kwargs: ...,
                    # priming_message: ...,
                    # auto_prime: ...,
                )
            ]
        )

    def chat(
        self,
        message: str,
        user_message_kwargs: dict = dict(),
        system_message_kwargs: dict = dict(),
    ) -> ChatMessage:
        """Chat with the assistant.

        This class allows for routing messages to certain actions.
        The default action is normal chatting with the assistant.
        """

        if message == "/save-system":
            return self.save_system_message()
        elif message.startswith("/save-steps"):
            return self.save_steps()
        else:
            return super().chat(
                message=message,
                user_message_kwargs=user_message_kwargs,
                system_message_kwargs=system_message_kwargs,
            )

    def save_system_message(self) -> ChatMessage:
        """Save last message from assistant as system message."""
        self._system_prompt = self.history[-1].message
        return {
            "role": "system",
            "message": "System message saved.",
        }

    def save_steps(self) -> ChatMessage:
        """Process and save steps given by assistant."""
        answer = self.chat(
            "Process the last message and return a structured list of strings "
            "containing each of the individual steps. "
            "The strings can have any kind of text and format. "
            "The important thing is to return a list of strings that can be "
            "evaluated with python using `eval`."
        )

        self._steps_prompts = eval(answer["message"])
