<!-- omit in toc -->

# Contributing to TSFX

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the
[Table of Contents](#table-of-contents) for different ways to help and details
about how this project handles them. Please make sure to read the relevant
section before making your contribution. It will make it a lot easier for us
maintainers and smooth out the experience for all involved. The community looks
forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's
> fine. There are other easy ways to support the project and show your
> appreciation, which we would also be very happy about:
>
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!--toc:start-->

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Legal Notice](#legal-notice)
  - [Reporting Bugs](#reporting-bugs)
    - [Before Submitting a Bug Report](#before-submitting-a-bug-report)
    - [How Do I Submit a Good Bug Report?](#how-do-i-submit-a-good-bug-report)
  - [Suggesting Enhancements](#suggesting-enhancements)
    - [Before Submitting an Enhancement](#before-submitting-an-enhancement)
    - [How Do I Submit a Good Enhancement Suggestion?](#how-do-i-submit-a-good-enhancement-suggestion)
  - [Your First Code Contribution](#your-first-code-contribution)
    - [Setting Up Your Development Environment](#setting-up-your-development-environment)
    - [Running the Tests and Checks](#running-the-tests-and-checks)
    - [Opening a Pull Request](#opening-a-pull-request)
  - [Improving The Documentation](#improving-the-documentation)

<!--toc:end-->
<!-- omit in toc -->

## I Have a Question

> If you want to ask a question, we assume that you have read the available
> [Documentation](https://wilswer.github.io/tsfx/).

Before you ask a question, it is best to search for existing
[Issues](https://github.com/wilswer/tsfx/issues) that might help you. In case
you have found a suitable issue and still need clarification, you can write your
question in this issue. It is also advisable to search the internet for answers
first.

If you then still feel the need to ask a question and need clarification, we
recommend the following:

- Open an [Issue](https://github.com/wilswer/tsfx/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (Python version, operating system,
  etc.), depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

<!-- omit in toc -->

> ### Legal Notice
>
> When contributing to this project, you must agree that you have authored 100%
> of the content, that you have the necessary rights to the content and that the
> content you contribute may be provided under the project license.

### Reporting Bugs

<!-- omit in toc -->

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more
information. Therefore, we ask you to investigate carefully, collect information
and describe the issue in detail in your report. Please complete the following
steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using
  incompatible environment components/versions (Make sure that you have read the
  [documentation](https://wilswer.github.io/tsfx/). If you are looking for
  support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the
  same issue you are having, check if there is not already a bug report existing
  for your bug or error in the
  [bug tracker](https://github.com/wilswer/tsfx/issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if
  users outside the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package
    manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with
    older versions?

<!-- omit in toc -->

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs
> including sensitive information to the issue tracker, or elsewhere in public.
> Instead sensitive bugs must be sent by email to <wilhelm.wermelin@icloud.com>.

We use GitHub issues to track bugs and errors. If you run into an issue with the
project:

- Open an [Issue](https://github.com/wilswer/tsfx/issues/new). (Since we can't
  be sure at this point whether it is a bug or not, we ask you not to talk about
  a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the _reproduction
  steps_ that someone else can follow to recreate the issue on their own. This
  usually includes your code. For good bug reports you should isolate the
  problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If
  there are no reproduction steps or no obvious way to reproduce the issue, the
  team will ask you for those steps and mark the issue as `needs-repro`. Bugs
  with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as
  well as possibly other tags (such as `critical`), and the issue will be left
  to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for TSFX,
**including completely new features and minor improvements to existing
functionality**. Following these guidelines will help maintainers and the
community to understand your suggestion and find related suggestions.

<!-- omit in toc -->

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://wilswer.github.io/tsfx/) carefully and find
  out if the functionality is already covered, maybe by an individual
  configuration.
- Perform a [search](https://github.com/wilswer/tsfx/issues) to see if the
  enhancement has already been suggested. If it has, add a comment to the
  existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's
  up to you to make a strong case to convince the project's developers of the
  merits of this feature. Keep in mind that we want features that will be useful
  to the majority of our users and not just a small subset. If you're just
  targeting a minority of users, consider writing an add-on/plugin library.

<!-- omit in toc -->

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as
[GitHub issues](https://github.com/wilswer/tsfx/issues).

- Use a **clear and descriptive title** for the issue to identify the
  suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many
  details as possible.
- **Describe the current behavior** and **explain which behavior you expected to
  see instead** and why. At this point you can also tell which alternatives do
  not work for you.
- **Explain why this enhancement would be useful** to most TSFX users. You may
  also want to point out the other projects that solved it better and which
  could serve as inspiration.

### Your First Code Contribution

`tsfx` is a hybrid Rust/Python project: the feature extractors are implemented
in Rust and exposed to Python through [PyO3](https://pyo3.rs/) and
[Maturin](https://maturin.rs/). To work on it locally you therefore need both a
Python and a Rust toolchain.

#### Setting Up Your Development Environment

**Prerequisites:**

- Python 3.9 or newer.
- A **nightly** Rust toolchain (the channel is pinned in `rust-toolchain.toml`).
  Install it via [rustup](https://rustup.rs/):
  ```bash
  rustup toolchain install nightly
  ```
  `rustup` automatically selects the nightly channel when you build inside the
  repository.

**Steps:**

1. Fork the repository, then clone your fork:
   ```bash
   git clone https://github.com/<your-username>/tsfx.git
   cd tsfx
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   The `dev` extra pulls in `maturin`, `ruff`, `mypy`, `pytest`, `pre-commit`,
   and the documentation tooling. To also install the benchmark and example
   dependencies, use `pip install -e ".[all]"` instead.

4. Compile the Rust extension into your environment:
   ```bash
   maturin develop
   ```
   Re-run this command whenever you change Rust code. Use
   `maturin develop --release` for an optimized build (recommended before
   running benchmarks).

5. Install the pre-commit hooks so formatting and linting run automatically on
   every commit:
   ```bash
   pre-commit install
   ```

#### Running the Tests and Checks

Before opening a pull request, make sure the test suite and linters pass. These
are the same checks enforced by the pre-commit hooks and CI.

Python tests (located in `python_tests/`):

```bash
pytest
```

Rust tests:

```bash
cargo test
```

Linting and formatting:

```bash
# Python
ruff check .
ruff format .
mypy

# Rust
cargo fmt
cargo clippy -- -D warnings
```

You can also run every pre-commit hook against the whole repository at once:

```bash
pre-commit run --all-files
```

#### Opening a Pull Request

1. Create a branch for your change:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes, adding tests and docstrings where appropriate.
3. Ensure all tests and checks pass (see above).
4. Commit your work — the pre-commit hooks will format and lint your changes.
5. Push your branch and open a pull request against the `main` branch,
   describing what you changed and why.

A maintainer will review your contribution as soon as possible. Thanks again! ❤️

### Improving The Documentation

Improving the documentation is one of the most valuable contributions you can
make. The documentation lives in a few places:

- **Python API reference** — generated from the
  [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html)
  docstrings in the package using
  [mkdocstrings](https://mkdocstrings.github.io/). To improve the API docs, edit
  the docstrings directly in the source.
- **Narrative docs** — the Markdown files under `docs/` (and `README.md`, which
  is embedded into the documentation home page). The site is built with
  [MkDocs](https://www.mkdocs.org/) and the
  [Material](https://squidfunk.github.io/mkdocs-material/) theme; configuration
  lives in `mkdocs.yml`.
- **Rust internals** — generated from the doc comments (`///`) in `src/` with
  `cargo doc`.

To preview the Python documentation locally (after running `maturin develop` so
the package is importable):

```bash
mkdocs serve
```

Then open <http://127.0.0.1:8000> in your browser; the site rebuilds
automatically as you edit.

To build the Rust documentation:

```bash
cargo doc --no-deps --document-private-items --open
```

The full documentation site is built and deployed to
[GitHub Pages](https://wilswer.github.io/tsfx/) automatically whenever a new
version tag is pushed (see `.github/workflows/docs.yml`), so you don't need to
deploy anything yourself — just include your documentation changes in your pull
request.

<!-- omit in toc -->

## Attribution

This guide is based on the [contributing.md](https://contributing.md/generator)!
