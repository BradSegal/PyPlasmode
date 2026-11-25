# Using PyPlasmode with a coding agent

A coding agent asked to benchmark feature selection can write a simulation that runs but answers the
wrong question. It may split the data after resampling, so that copies of a row reach both training
and test; read recovery without comparing it with chance; or report a single simulated dataset. The
[`skills/pyplasmode`](https://github.com/BradSegal/PyPlasmode/tree/main/skills/pyplasmode) skill
gives the agent the workflow and the checks that prevent these mistakes.

The skill follows the open skill layout, a `SKILL.md` file and a `references` folder of complete
recipes. If your coding agent supports this format, copy the folder into its skill directory, so
that the agent loads the skill whenever a task matches its description. Otherwise, ask an agent with
file access to read `SKILL.md` and its recipes before it starts.

The recipes cover comparing two procedures, group recovery, sample size, censored
time-to-event outcomes and target AUCs. The test suite runs every Python example in the skill
against the library, so the recipes stay correct as the library changes.

Generated matrices contain copies of source rows, so the
[data handling guidance](privacy-and-limitations.md) applies to anything an agent writes to disk.
