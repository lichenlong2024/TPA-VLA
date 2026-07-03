# Release Audit

This package is prepared as a reviewer-facing TPA-VLA code release. It excludes:

- unpublished continual-learning/routing experiments,
- alternative modules not used by the AAAI paper,
- raw datasets, videos, checkpoints, and logs,
- local machine paths and cloud instance paths,
- credentials or private configuration files.

Before pushing, run:

```powershell
python public_release_tpa_vla/scripts/audit_release.py public_release_tpa_vla
rg --files public_release_tpa_vla
```

Expected result: the audit command should pass and the file list should contain
only source code, documentation, and lightweight configuration files.
