"""No-op analyze step.

``launch_cells.sh`` always chains an ``afterany`` analyze job.  The #2847 study's
analysis is cross-arm and runs once by hand after all four arms drain, so each
arm points ``CALIB_ANALYZE`` here rather than at an analyzer that would run four
times on a quarter of the data each.
"""

print("noop analyze: #2847 analysis is cross-arm; run analyze_spikes.py once all arms drain")
