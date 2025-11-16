# CyclopsHGT

## Goal

Build an HGT on ROBOKOP and maybe try out SAE it.

## Basic Setup

* github: This project has a github repo at https://github.com/cbizon/CyclopsHGT
* uv: we are using uv for package and environment management and an isolated environment
* tests: we are using pytest, and want to maintain high code coverage

### Environment Management - CRITICAL
**NEVER EVER INSTALL ANYTHING INTO SYSTEM LIBRARIES OR ANACONDA BASE ENVIRONMENT**
- ALWAYS use the isolated virtual environment at `.venv/`
- ALWAYS use `uv run` to execute commands, which automatically uses the isolated environment
- The virtual environment is sacred. System packages are not your garbage dump.

## Key Dependencies

DGL: Deep Graph Library

## Basic Workflow

## Input

The input data may never be changed.  We are working with a filtered version of the ROBOKOP Knowledge Graph, a large biomedical KG with strong types and heterogenous relationships.  Our initially filtered version is about 0.5M nodes and 8M edges.

## Project structure
input/
src/
tests/

## ***RULES OF THE ROAD***

- CLEAN CLEAN CLEAN - no extra junk, no leftover files, no dead code

- Keep docs up to date, even (especially) if I don't ask you to

- README is for humans, CLAUDE is for machines

- Don't use mocks. They obscure problems

- Ask clarifying questions

- Don't make classes just to group code. It is non-pythonic and hard to test.

- Do not implement bandaids - treat the root cause of problems

- Don't use try/except as a way to hide problems.  It is often good just to let something fail and figure out why.

- Once we have a test, do not delete it without explicit permission.  

- Do not return made up results if an API fails.  Let it fail.

- When changing code, don't make duplicate functions - just change the function. We can always roll back changes if needed.

- Keep the directories clean, don't leave a bunch of junk laying around.

- When making pull requests, NEVER ever mention a `co-authored-by` or similar aspects. In particular, never mention the tool used to create the commit message or PR.

- Check git status before commits

