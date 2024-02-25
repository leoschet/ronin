# Ronin
Rogue assistant that follows its own path, with no lord or master.

## Setup

- Create a new Python environment.
  - Check the version requirements in [pyproject.toml](./app/backend/pyproject.toml).
- This project uses poetry for dependency management. Install it with `pip install poetry`.
- By default, this project uses `ruff` for linting and code formatting.
  - For some reason, `ruff` doesn't work with poetry, so install it with `pip install ruff`.

## Extending functionality

Ronin provides base functionality for chat based applications. More specific functionality can be added to Ronin via plugins.

### Plugins

A plugin is essentially a python module that expands on base functionality from Ronin. With a plugin we can implement new assistants, or any other component defined in the Ronin codebase.

Plugins that rely on the Register design pattern (i.e.: `assistants`) must be registered as a `"Ronin.plugins"` entry point in the project configuration file (normally `pyproject.toml`).

```toml
[tool.poetry.plugins."ronin.plugins"]
"prompts" = "myapp.assistants"
```

Ronin loads plugins in the `__init__.py` file at the root of the project.

#### Reference:
- [Advertising behavior](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#advertising-behavior).
- [About poetry and application plugins](https://python-poetry.org/docs/master/plugins/)
- [Converting "entry_points" from setup.py to pyproject.toml](https://github.com/python-poetry/poetry/issues/658#issue-382935272)