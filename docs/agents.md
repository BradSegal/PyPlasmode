# Using PyPlasmode with a coding agent

A coding agent asked to benchmark feature selection can write a simulation that runs but answers
the wrong question. It may split the data after resampling, so that copies of a row reach both
training and test; read recovery without comparing it with chance; or report a single simulated
dataset. The [`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode)
skill gives the agent the workflow and the checks that prevent these mistakes: how to choose a
signal structure and an outcome, how to read the results, what goes wrong and what to report.

The skill follows the open skill layout: a `SKILL.md` file with a name, a description of when to
use it and the instructions, and a `references` folder of complete recipes. If your coding agent
supports this format, copy the folder into its skill directory, so that the agent loads the
skill whenever a task matches its description. Otherwise, ask an agent with file access to read
`SKILL.md` and its recipes before it starts.

The recipes cover repeating a study and comparing two procedures, correlated features with group
recovery, how recovery varies with sample size, time-to-event outcomes with censoring, and
setting difficulty with a target AUC. The test suite runs every Python example in the skill
against the library, so the recipes stay correct as the library changes.

Generated matrices contain copies of source rows, so the
[data handling guidance](privacy-and-limitations.md) applies to anything an agent writes to disk.
