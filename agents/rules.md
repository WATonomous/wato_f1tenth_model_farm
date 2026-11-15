# Why the hard rules exist

1. **The Data Standard is the contract.** Extraction, labelling, review and training are written by
   different members at different times. They only fit together if every file matches one
   definition, and one library writes it. A silent "fix" in code splits the data into two formats.
2. **No data in git.** Bags and datasets are gigabytes, and recordings can show people. Once
   pushed they stay in history.
3. **Dataset files are permanent.** Sample IDs are referenced by labels, masks, `tracks.json`,
   predictions and CVAT tasks. Renaming or renumbering breaks every one of those links. Reviewed
   and corrected annotations are hours of members' work; a generator re-run must not erase them.
4. **No overwritten outputs.** A half-written file looks valid until it is read. Old outputs are
   what new results are compared against.
5. **Ask before GPU work.** The laptop GPU and WATcloud's GPUs are shared and limited; a job left
   running blocks other members (WATcloud does not enforce GPU-memory slices).
6. **One Python, declared dependencies.** Python 3.12 matches ROS 2 Jazzy, so code moves into the
   car stack. An env changed by hand cannot be rebuilt on WATcloud or another laptop.
7. **Tests after every change.** Most bugs here corrupt data quietly; the CPU tests are the
   cheapest place to catch them.
8. **Keep to the task.** Extraction and training belong to members who are learning by building
   them; doing their task for them takes that away.
9. **No agent credit in git.** The team's rule, the same as in the car stack: commits show the
   member who made them.
