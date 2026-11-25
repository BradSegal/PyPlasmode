# Using PyPlasmode with a coding agent

The repo includes a skill package for teaching a coding agent how to design, run, and read a
plasmode simulation with PyPlasmode. It covers the order of steps, which signal structure and
recovery measure are appropriate for a question, and pitfalls that lead to misleading
simulations.

The package is the folder
[`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode).
It follows the open skill layout: a `SKILL.md` file with a name, a description of when to use
it and the instructions, and a `references` folder with complete recipes. If your coding agent
supports this skill format, copy the folder to its skill directory. You can also ask an agent
with file access to read `SKILL.md` and the recipes it points to. Running the examples needs a
supported Python version with PyPlasmode installed.

Recipes cover repeating a study and comparing two procedures, correlated features with group
recovery, how recovery varies with sample size, time-to-event outcomes with censoring, and
calibrating difficulty with a target AUC. The test suite runs the Python examples against the
public API.

The skill uses the same terms as this documentation. Simulated datasets include source rows, so
the [data handling guidance](privacy-and-limitations.md) applies to anything an agent writes to
disk.
