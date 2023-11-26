import attrs

from ronin.assistants.base import (
    AssistantRegister,
    ChatAssistant,
    ProactiveChatAssistant,
)


@AssistantRegister.register("resume-bio-writer")
@attrs.define
class ResumeBioWriter(ChatAssistant):
    priming_message: str | None = (
        "Write the top principles and concepts that can make "
        "someone stand out in Linkedin."
    )
    auto_prime: bool = True


@AssistantRegister.register("resume-experience-writer")
@attrs.define
class ResumeExperienceWriter(ProactiveChatAssistant):
    priming_message: str | None = (
        "Write the top principles and concepts one should take into account "
        "when writing about their past professional experiences in their resume."
    )
    auto_prime: bool = True
