# Codex Notify

This folder contains the repository-scoped Codex notification script.

## Files

- [codex_notify.py](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/scripts/notify/codex_notify.py)
  - notification hook target for Codex
- [test_notify.py](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/scripts/notify/test_notify.py)
  - sends a synthetic `agent-turn-complete` event for manual verification

## Local Machine Setup

The repository script expects local Pushover credentials in:

- `~/.codex/notify_secrets.toml`

Example:

```toml
[pushover]
app_token = "YOUR_APP_TOKEN"
user_key = "YOUR_USER_KEY"
```

Environment variables are also supported:

- `PUSHOVER_APP_TOKEN`
- `PUSHOVER_USER_KEY`

## Codex Config

Your machine-level Codex config must point the notify hook at this script.
On Windows, the relevant entry is in:

- `C:\Users\<you>\.codex\config.toml`

Example:

```toml
[windows]
notify = ["python", "C:\\path\\to\\repo\\scripts\\notify\\codex_notify.py"]
```

## Portability

On another computer, the repo copy alone is not enough.
You also need:

1. a valid `~/.codex/notify_secrets.toml`
2. a `~/.codex/config.toml` notify hook that points to the clone location

After setup, run [test_notify.py](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/scripts/notify/test_notify.py) once to verify delivery.
