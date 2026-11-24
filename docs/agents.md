# Using PyPlasmode with a coding agent

The repository includes a skill package that teaches a coding agent how to design, run and
read a plasmode simulation with PyPlasmode: the order of the steps, which signal structure
and recovery measure fit a question, and the mistakes that make a simulation misleading.

The package is the folder
[`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode).
It follows the open skill layout: a `SKILL.md` file with a name, a description of when to use
it and the instructions, and a `references` folder with complete recipes. Copy the folder to
the place your agent loads skills from, or point the agent at `SKILL.md` directly. It needs
no tools beyond Python with PyPlasmode installed.

The recipes cover repeating a study and comparing two methods, correlated features with
group recovery, sizing a study by its number of events, time-to-event outcomes with
censoring, and setting difficulty with a target AUC. Every code block in the package is run
by the test suite against the current API.

The skill uses the same terms as this documentation. Generated datasets contain rows of the
source data, so the [data handling guidance](privacy-and-limitations.md) applies to anything
an agent writes to disk.
