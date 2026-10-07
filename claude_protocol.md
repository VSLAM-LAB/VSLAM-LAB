We're going to work on VSLAM-LAB in iterative steps, following a strict protocol.

## Protocol

### Step 1: Task
I tell you what we are implementing. Don't start changing anything yet.

### Step 2: Design discussion
We discuss the details of the implementation. Explore the relevant code, ask questions, propose an approach, and point out trade-offs or risks. Do not move to Step 3 until I explicitly confirm the design.

### Step 3: Implementation (debugging loop)
Once I confirm, implement it as follows:

  1. Define success: write down concrete, checkable criteria (e.g. the run completes without errors, expected output files exist, specific log lines or metrics appear). Share them briefly.
  2. Loop until the success criteria are met:
     - Make changes
     - Run: pixi run vslamlab configs/exp_debug.yaml --overwrite
       (while it runs, monitor swap usage; see "Swap monitoring" below)
     - Inspect the output and logs, then compare against the success criteria
     - If it fails, diagnose the root cause and iterate. Don't paper over errors.
  3. When success is reached, tell me the implementation is finished and summarize what changed and what you verified.
  4. I will run the full experiment myself and report back.
     - If I say "keep", the implementation is accepted and you move to Step 4.
     - If I give feedback or report problems, go back to the start of the loop with new success criteria derived from my feedback.

### Step 4: Commit
  1. Design the commit(s): propose the split, the files in each, and the commit messages.
  2. Tell me about the proposed commits and wait for my answer. Do not commit before I approve.
  3. After I approve, commit. Then we're ready for the next task (back to Step 1).

## Swap monitoring
Swap usage must be monitored whenever you run anything heavy (including the debug experiment).

- Check swap usage (e.g. `free -m` or `swapon --show`) before starting a run and periodically while it runs.
- If swap usage rises above 80%:
  1. Stop immediately. Kill the running process and don't launch anything new.
  2. Tell me the swap level you observed and what was running.
  3. Wait. I will clean up swap/memory myself. Don't try to clear it, restart services, or work around it.
  4. Do not resume until I explicitly tell you to. When I do, pick up exactly where you left off: same step, same success criteria, same point in the loop. Don't restart the task or redefine success unless I ask.
- After resuming, keep monitoring as before.

## Rules
- Never skip a step or advance without my confirmation where one is required (end of Step 2, "keep" in Step 3, approval in Step 4).
- Keep changes minimal and focused on the agreed task. Don't refactor unrelated code.
- Always use --overwrite with the debug config so runs are reproducible.
- If something in the task is ambiguous, ask during Step 2 rather than guessing during Step 3.
- Be explicit about what you ran and what you observed, so I can verify.
- Swap above 80% is a hard stop: halt, report, and wait for my explicit go-ahead to resume (see "Swap monitoring").

Acknowledge this protocol, then wait for my first task.
