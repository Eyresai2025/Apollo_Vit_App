# Apollo VIT Central Configuration

## Purpose

All application configuration is now loaded through `src/COMMON/config.py`.
The existing `.env` remains the deployment configuration file, so current
PLC, camera, recipe and UI modules continue to work.

## Configuration precedence

1. Operating-system environment variable
2. Project `.env`
3. Typed default in `src/COMMON/config.py`

## New typed usage

```python
from src.COMMON.config import get_config

config = get_config()
print(config.database.name)
print(config.plc.ip)
print(config.paths.capture_dir)
print(config.camera.serials["sidewall1"])
```

## Existing usage remains supported

```python
from src.COMMON.common import load_env

env = load_env()
print(env["PLC_IP"])
```

`load_env()` now delegates to the central service. This allows modules to be
migrated one at a time.

## Validate configuration

```bat
python tools\validate_configuration.py
```

Write a masked JSON snapshot:

```bat
python tools\validate_configuration.py --json media\validation_reports\configuration_snapshot.json
```

The validator does not connect to the PLC, cameras, lasers or MongoDB. It
checks types, paths, PLC addresses, duplicate camera serials, recipe targets,
model paths and required values.

## Main sections

- `config.application`
- `config.database`
- `config.paths`
- `config.models`
- `config.inference`
- `config.plc`
- `config.health`
- `config.camera`
- `config.devices`
- `config.recipe`

## Reload after an approved settings change

```python
from src.COMMON.config import reload_config

config = reload_config()
```

Do not reload configuration during an active inspection cycle. Apply changes
when the application is idle.
