# Contributing to u-Unwrap3D

Thank you for your interest in contributing to u-Unwrap3D. Contributions can include bug reports, feature requests, documentation improvements, tutorials, tests, examples, and code changes.

Please follow the project [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

## Reporting Issues

Before opening a new issue, please check whether a similar issue already exists. When reporting a bug, include:

- A short description of the problem and expected behavior.
- The u-Unwrap3D version or commit used.
- Operating system, Python version, and installation method.
- Relevant package versions, especially mesh, geometry, image-processing, or visualization dependencies.
- A minimal example, command, script, screenshot, or error log that helps reproduce the issue.
- Whether the issue occurs during image preprocessing, segmentation, mesh generation, surface mapping, visualization, or downstream analysis.

Please do not upload large or private microscopy datasets directly to GitHub. Use a small reproducible example when possible, or describe how maintainers can access appropriate test data.

## Development Setup

A typical local development setup is:

```bash
git clone https://github.com/DanuserLab/u-unwrap3D.git
cd u-unwrap3D
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Some dependencies used for geometry and mesh processing may be platform-specific. If a contribution affects those functions, please document the operating system and package versions used for testing.

## Pull Requests

For code or documentation changes:

- Create a focused branch for the change.
- Keep pull requests small and specific when possible.
- Describe what changed, why it changed, and how it was tested.
- Preserve existing public functions and module organization unless a breaking change is discussed first.
- Update documentation, tutorials, or parameter descriptions when user-facing behavior changes.
- Avoid committing generated files, large datasets, local paths, credentials, or machine-specific configuration.

## Testing and Validation

Before submitting a pull request, run the most relevant checks for the change. At minimum, verify that the package imports:

```bash
python -c "import unwrap3D; print(unwrap3D.__file__)"
```

For changes affecting workflows, run a small example using data from `example_data` when possible. For documentation changes, check that links, commands, and example paths are accurate.

If a full test cannot be run because it requires large data, specialized dependencies, or specific hardware, explain what was tested and what remains untested in the pull request.

## Coding and Documentation Guidelines

- Follow the style and structure of the surrounding code.
- Prefer clear, maintainable implementations over large rewrites.
- Keep algorithmic behavior stable unless the change is intentional and documented.
- Add comments only where they clarify non-obvious logic.
- Use small example data or synthetic data for tests and tutorials whenever possible.
- Document new parameters, expected inputs, outputs, and limitations.

## License and Citation

u-Unwrap3D is distributed under the GNU General Public License v3.0. By contributing, you agree that your contribution will be distributed under the same license.

If you use u-Unwrap3D in scientific work, please cite the associated publication or preprint listed in the README.
