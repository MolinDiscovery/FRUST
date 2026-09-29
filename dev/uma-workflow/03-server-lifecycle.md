# 03 — Reuse one UMA server per submitted job

## Goal

Start one UMA server inside each submitted target job, reuse it for all UMA
stages and numerical-frequency calls in that job, and stop it on completion or
failure. The ORCA clients and server must run on the allocated compute node.

## Current behavior to replace

`Stepper.orca(..., uma=...)` currently owns a server for one method call. A
workflow with separate UMA SP and optimization stages therefore starts and
stops the server between stages. Keep standalone `Stepper` calls usable while
adding a wider, explicit lifetime for workflow jobs.

## Work

1. Design a job-scoped UMA server context that workflow execution can share
   across stages. Route ORCA client inputs to its loopback address. Do not
   start a server from the login process for a submitted job.
2. Ensure that `NumFreq` displacements in one ORCA job use the same server.
   Detect and prevent duplicate live servers for the same target job.
3. Ensure cleanup after success, stage failure, exception, and cancellation
   where process signals allow it. Preserve useful failure logs.
4. On node066 in `kemi1`, run a focused SP → optimization → small `NumFreq`
   sequence inside one allocation. Record hostnames, process IDs, server
   start/stop events, and peak number of UMA servers.

## Acceptance

- Exactly one UMA server process is live for the target job while its UMA
  stages run; repeated client requests reuse the same server process ID.
- Client and server hostnames match the allocated compute node. The bind
  address is local to that node.
- After the job exits, no UMA server from that job remains.
- Standalone UMA `Stepper` use and non-UMA workflows still work.

## Completion record

Completed 2026-09-29. FRUST revision `3aebf43` contains the job-scoped server
implementation tested on the cluster. The workflow runner now opens a lazy UMA
server scope **inside each executing target job or stage-group job**. The first
UMA stage starts the server; subsequent UMA stages, including `NumFreq`, reuse
its PID and loopback bind. A standalone `Stepper.orca(..., uma=...)` call still
owns its own server. A group with no UMA stages creates no server.

The runtime can be pinned per workflow run or submission:

```python
jobs = wf.submit(
    out_dir="results/uma_screen",
    cluster=cluster,
    uma_oet_tools="/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu",
)
```

This path is selected inside each executing job. The login process only
submits the target jobs. FRUST also preserves an explicit `OET_TOOLS` setting
when loading `.env` and reapplies the selected runtime after `Stepper` imports
its calculator dependencies. All UMA stages in one job must use the same server
resources and log policy; a conflicting request fails instead of starting a
second server. The server exits in the stage-group `finally` path after normal
completion or a Python exception. A SIGTERM handler unwinds that scope, which
was checked in a subprocess test. SIGKILL cannot run Python cleanup; Slurm's
job process cleanup still applies.

### Compute-node result

The [submitted water workflow](evidence/task03/submit_lifecycle.py) ran UMA
single point → optimization → numerical frequencies through `wf.submit()` as
one target job. Its ORCA inputs selected `omol@uma-s-1p2p1` with
GFN2-xTB ALPB(chloroform). The [Slurm record](evidence/task03/run-c/slurm_status.txt)
shows job **65678046** completed on `node066` in `kemi1`. The full run artifacts
remain at
`/lustre/hpc/kemi/jmni/results/uma-task03-lifecycle-20260929-c`.

| Check | Observed result |
| --- | --- |
| Submission host | `fend05.cluster`; no UMA server was started during submission |
| Server | PID `700314` on `node066.cluster`, bound to `127.0.0.1:51363` |
| UMA stage PIDs | `uma_sp`, `uma_opt`, and `uma_numfreq` all recorded server PID `700314` |
| Clients | 21 external requests, all from `node066.cluster` to `127.0.0.1:51363` |
| Server lifetime | One start, one stop, peak live server count 1, none remaining after the job |
| Calculations | All three stages normally terminated; `NumFreq` returned three water frequencies |

The [audit summary](evidence/task03/run-c/lifecycle_summary.json) checks the
stage PIDs, client and server hostnames, bind addresses, live-server count,
normal termination, and returned frequencies. The
[client calls](evidence/task03/run-c/client_calls.log),
[server log](evidence/task03/run-c/oet_uma_server_51363.log), and
[Submitit log](evidence/task03/run-c/submitit.out) preserve the underlying
events. The client audit used a temporary wrapper that logged hostname and
then executed the pinned OET client's binary; it did not change the runtime's
calculator code.

After completion, a [separate node066 allocation](evidence/task03/run-c/post_job_process_check.txt)
(job `65678076`) checked PID `700314` and reported `server_absent`.

The first cluster attempt, job `65678010`, revealed that OET startup on a cold
cluster filesystem can exceed the previous 120-second readiness limit. FRUST
now allows up to 600 seconds, while still failing promptly if the server
process exits. A second successful run, job `65678028`, verified the
calculations and one-server lifetime; its client audit file was missing because
the temporary wrapper relied on an environment variable that Slurm did not
export. Job `65678046` used a fixed audit path and satisfied every check.

The full fast test suite passed in the `UMA` environment: **353 passed,
13 deselected**. The three installed-OET slow tests also passed. A dedicated
SIGTERM test verified that a server started by a job scope is stopped and its
log records both events. A high-level catalyst-screen submission test confirms
that `uma_oet_tools` reaches each child target job.
