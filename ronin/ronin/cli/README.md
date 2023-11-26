# Ronin CLI interface

The `--help` flag can be passed to any command to get a complete overview of all its options.

## Chat command

```
Usage: ronin chat [OPTIONS]

  Chat with Ronin assistants.

Options:
  -a, --assistant TEXT    ID of the assistant you want to chat with.
  -m, --message TEXT      Initial message to send to assistant.
  --system-message TEXT   System message to use. This overrides the existing
                          system message.
  --max-length INTEGER    Maximum length of the response.
  -i, --interactive       Whether to keep the conversation open or not.
  -o, --output-path TEXT  Path where to save .json file with output.
  --help                  Show this message and exit.
```

The `chat` command is used to interact with the Ronin assistants.

Example:
```
poetry run ronin chat -i -m hello --system-message "Helpful creature, you are. Speak like yoda, you will."
```