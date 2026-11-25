# Using PyPlasmode with a coding agent

Use [`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode)
to guide a coding agent through simulation design, execution and interpretation, including
signal and recovery choices and common pitfalls.
It follows the open skill layout: a `SKILL.md` file with a name, a description of when to use
it and the instructions, and a `references` folder with complete recipes. If your coding agent
supports this skill format, copy the folder to its skill directory. With an agent that has file access, ask it to read `SKILL.md` and its linked recipes. Running the examples needs a
supported Python version with PyPlasmode installed.

Recipes cover repeating a study and comparing two procedures, correlated features with group
recovery, how recovery varies with sample size, time-to-event outcomes with censoring, and
calibrating difficulty with a target AUC. The test suite runs the Python examples against the
public API.

Simulated datasets include source rows, so
the [data handling guidance](privacy-and-limitations.md) applies to anything an agent writes to
disk.
