# Using PyPlasmode with a coding agent

The repository includes a skill package that teaches a coding agent how to design, run and
read a plasmode simulation with PyPlasmode: the order of the steps, which signal structure
and recovery measure fit a question, and the mistakes that make a simulation misleading.

The package is the folder
[`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode).
It follows the open skill layout: a `SKILL.md` file with a name, a description of when to use
it and the instructions, and a `references` folder with complete recipes. If your coding agent
supports this skill format, copy the folder to its skill directory. You can also ask an agent
with file access to read `SKILL.md` and the recipes it points to. Running the examples needs a
supported Python version with PyPlasmode installed.

The recipes cover repeating a study and comparing two procedures, correlated features with
group recovery, how recovery varies with sample size, time-to-event outcomes with censoring,
and setting difficulty with a target AUC. The test suite runs the Python examples against the
public API.

The skill uses the same terms as this documentation. Generated datasets contain rows of the
source data, so the [data handling guidance](privacy-and-limitations.md) applies to anything
an agent writes to disk.
