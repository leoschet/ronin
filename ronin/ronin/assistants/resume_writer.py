import attrs

from ronin.assistants.chat import ChatAssistant, ProactiveChatAssistant


@attrs.define
class ResumeBioWriter(ChatAssistant):
    def prime(self):
        self.chat(
            "Write the top principles and concepts that can make someone stand out in Linkedin."
        )


@attrs.define
class ResumeExperienceWriter(ProactiveChatAssistant):
    def prime(self):
        self.chat(
            (
                "Write the top principles and concepts one should take into account "
                "when writing about their past professional experiences in their resume."
            )
        )
