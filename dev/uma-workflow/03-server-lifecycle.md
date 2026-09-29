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

Pending. Add the FRUST revision, node066 job ID, process/hostname evidence,
test results, and any lifecycle limitation here.
