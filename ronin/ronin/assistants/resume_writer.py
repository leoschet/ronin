import attrs

from ronin.assistants.base import ChatAssistant, ProactiveChatAssistant


@attrs.define
class ResumeBioWriter(ChatAssistant):
    priming_message: str | None = (
        "Write the top principles and concepts that can make "
        "someone stand out in Linkedin."
    )
    auto_prime: bool = True


@attrs.define
class ResumeExperienceWriter(ProactiveChatAssistant):
    priming_message: str | None = (
        "Write the top principles and concepts one should take into account "
        "when writing about their past professional experiences in their resume."
    )
    auto_prime: bool = True
