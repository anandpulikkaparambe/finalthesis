# Real-hardware deployment: 16 runs of the headline checkpoint

Raw ROS logs (`run_01` to `run_16`), the session report (`REPORT.md`) and a field log
(`field_log.html`), unedited. The checkpoint is `../headline_run/ur5e_isaac_sac_final.zip`; the script is
`../../deployment/real_phase2_unclamped_run.py`.

`REPORT.md` has the run-by-run table. Runs 1 (observation-only dry run) and 13 (target constant corrupted
by a joint-state merge bug, data discarded) are not analysable. Best approach 0.407 to 0.412 m; run 16
ended with a visually confirmed self-collision.