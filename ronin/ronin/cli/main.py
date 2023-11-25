import click

from ronin.cli.chat import chat

@click.group()
def ronin():
    """Ronin cli."""
    pass

ronin.add_command(chat)