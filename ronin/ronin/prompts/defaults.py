from ronin.prompts.templates import UserMessageTemplate

# XXX: We should evaluate whether we want to keep a large file full of templates,
# or if we want to manage defaults scattered across the codebase.
DEFAULT_PROACTIVE_MESSAGE_TRIGGER = UserMessageTemplate.from_str(
    prompt=(
        "Generate 5 questions to the user that will help you get all the information "
        "that is relevant for you to fulfill your task."
    )
)
