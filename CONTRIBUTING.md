# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

### Types of Contributions

#### Report bugs

Report bugs at https://github.com/chuber1986/ood-robustness.git/issues.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

#### Fix Bugs

Look through the GitLab issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

#### Implement Features

Look through the GitLab issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

#### Write Documentation

ood_robustness could always use more documentation, whether as part of the official ood_robustness docs, in docstrings, or even on the web in blog posts, articles, and such.

#### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/chuber1986/ood-robustness.git/issues.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up ood-robustness for local development.

1. Fork the ood-robustness repo on GitLab.
2. Clone your fork locally (if possible use the SSH link, starting with "git://" instead of hte HTTPS link starting with "https://"):
    ```bash
    $ git clone --recursive https://github.com/chuber1986/ood-robustness.git
    $ cd ood-robustness
    ```
3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:
    ```bash
    $ conda create -n ood-robustnesss python=3.11
    $ conda env update -n ood-robustness --file environment.yaml
    ```
    or use:
    ```bash
    $ make environment
    ```
4. Install pre-commit hook:
   Once after clone the replository run:

    ```bash
    $ pre-commit install
    ```

5. Create a branch for local development:

    ```bash
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the tests, including testing other Python versions with tox:

    ```bash
    $ python setup.py pytest  # test using the current environment
    $ pytest                  # alternative command
    $ tox                     # test on multiple environments
    ```

    To get flake8 and tox, just pip install them into your virtualenv.

7. Commit your changes and push your branch to GitLab:

    ```bash
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

8. Submit a pull request through the GitLab website.

## Merge Request Guidelines

Before you submit a merge request, check that it meets these guidelines:

1. The merge request should include tests.
2. If the merge request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
3. The merge request should work for Python 3.9, 3.10, and for PyPy. Check https://github.com/chuber1986/ood-robustness.git/merge_requests and make sure that the tests pass for all supported Python versions.

## Tips

To run a subset of tests:

```bash
$ pytest tests.test_ood_robustness
```

## Deploying

A reminder for the maintainers on how to deploy. Make sure all your changes are committed (including an entry in HISTORY.rst). Then run:

```bash
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```
