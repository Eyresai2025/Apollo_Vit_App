# src/COMMON/plc_result_sender.py

from pathlib import Path


def _project_root():
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _load_env(env_path=None):
    env_file = Path(env_path) if env_path else (_project_root() / ".env")
    data = {}

    try:
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass

    return data


def _env_int(env, key, default):
    try:
        value = env.get(key, "")
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _read_bit(client, db, byte, bit):
    data = client.db_read(db, byte, 1)
    if not data:
        return None
    return bool(data[0] & (1 << bit))


def _write_bit(client, db, byte, bit, value):
    data = bytearray(client.db_read(db, byte, 1))
    if not data:
        data = bytearray([0])

    if value:
        data[0] = data[0] | (1 << bit)
    else:
        data[0] = data[0] & ~(1 << bit)

    client.db_write(db, byte, data)

    try:
        return _read_bit(client, db, byte, bit)
    except Exception:
        return None


def send_tyre_result_to_plc(final_result, env_path=None):
    """
    Sends final tyre result to PLC using existing PLC client from full_hardware_check.

    OK/PASS  -> ACCEPT bit ON, REJECT bit OFF
    NG/DEFECT/SUSPECT/INVALID/FAILED -> ACCEPT bit OFF, REJECT bit ON

    DEPLOYMENT=False:
        no PLC write, returns Demo message.
    """

    env = _load_env(env_path)

    deployment = str(env.get("DEPLOYMENT", "False")).strip()

    if deployment != "True":
        return {
            "sent": False,
            "display": "Demo - Not Sent",
            "detail": "DEPLOYMENT=False",
        }

    final_result = str(final_result or "").strip().upper()

    if final_result in ("WAITING", "-", "", "UNKNOWN"):
        return {
            "sent": False,
            "display": "Not Sent",
            "detail": "No final result available",
        }

    accept_db = _env_int(env, "PLC_ACCEPT_DB", 100)
    accept_byte = _env_int(env, "PLC_ACCEPT_BYTE", 0)
    accept_bit = _env_int(env, "PLC_ACCEPT_BIT", 2)

    reject_db = _env_int(env, "PLC_REJECT_DB", 100)
    reject_byte = _env_int(env, "PLC_REJECT_BYTE", 0)
    reject_bit = _env_int(env, "PLC_REJECT_BIT", 3)

    try:
        from src.COMMON.full_hardware_check import get_hardware_state

        state = get_hardware_state()
        client = state.get("plc_client")

        if client is None:
            return {
                "sent": False,
                "display": "PLC Not Connected",
                "detail": "No PLC client available",
            }

        is_accept = final_result in ("OK", "PASS", "GOOD")
        is_reject = final_result in ("NG", "DEFECT", "SUSPECT", "INVALID", "FAILED", "FAIL")

        if is_accept:
            accept_readback = _write_bit(client, accept_db, accept_byte, accept_bit, True)
            reject_readback = _write_bit(client, reject_db, reject_byte, reject_bit, False)

            return {
                "sent": bool(accept_readback is True),
                "display": "ACCEPT Sent" if accept_readback is True else "ACCEPT Write Failed",
                "detail": f"ACCEPT={accept_readback}, REJECT={reject_readback}",
            }

        if is_reject:
            accept_readback = _write_bit(client, accept_db, accept_byte, accept_bit, False)
            reject_readback = _write_bit(client, reject_db, reject_byte, reject_bit, True)

            return {
                "sent": bool(reject_readback is True),
                "display": "REJECT Sent" if reject_readback is True else "REJECT Write Failed",
                "detail": f"ACCEPT={accept_readback}, REJECT={reject_readback}",
            }

        return {
            "sent": False,
            "display": "Result Not Mapped",
            "detail": f"Unknown final_result={final_result}",
        }

    except Exception as e:
        return {
            "sent": False,
            "display": "PLC Send Failed",
            "detail": str(e),
        }