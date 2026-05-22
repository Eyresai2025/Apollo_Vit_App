# src/COMMON/plc_result_sender.py

from pathlib import Path
import time


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


def _connect_plc(env):
    try:
        import snap7
    except Exception as e:
        raise RuntimeError(f"snap7 import failed: {e}")

    plc_ip = env.get("PLC_IP", "192.168.10.1")
    rack = _env_int(env, "PLC_RACK", 0)
    slot = _env_int(env, "PLC_SLOT", 1)

    client = snap7.client.Client()
    client.connect(plc_ip, rack, slot)

    if hasattr(client, "get_connected"):
        if not client.get_connected():
            raise RuntimeError(f"PLC connect failed: {plc_ip}")

    return client


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
    Sends final tyre result to PLC using a dedicated PLC connection.

    OK/PASS/GOOD/ACCEPT:
        ACCEPT bit pulse

    NG/DEFECT/SUSPECT/INVALID/FAILED/FAIL/REJECT:
        REJECT bit pulse

    This avoids using the shared Test Mode PLC client during Live.
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

    pulse_ms = _env_int(env, "PLC_RESULT_PULSE_MS", 300)

    is_accept = final_result in (
        "OK",
        "PASS",
        "GOOD",
        "ACCEPT",
    )

    is_reject = final_result in (
        "NG",
        "DEFECT",
        "SUSPECT",
        "INVALID",
        "FAILED",
        "FAIL",
        "REJECT",
    )

    if not is_accept and not is_reject:
        return {
            "sent": False,
            "display": "Result Not Mapped",
            "detail": f"Unknown final_result={final_result}",
        }

    client = None

    try:
        client = _connect_plc(env)

        # Clear both result bits before sending new result.
        _write_bit(client, accept_db, accept_byte, accept_bit, False)
        _write_bit(client, reject_db, reject_byte, reject_bit, False)

        time.sleep(0.05)

        if is_accept:
            readback = _write_bit(
                client,
                accept_db,
                accept_byte,
                accept_bit,
                True,
            )

            if pulse_ms > 0:
                time.sleep(pulse_ms / 1000.0)
                _write_bit(
                    client,
                    accept_db,
                    accept_byte,
                    accept_bit,
                    False,
                )

            return {
                "sent": readback is True,
                "display": "ACCEPT Sent" if readback is True else "ACCEPT Write Failed",
                "detail": f"ACCEPT readback={readback}",
            }

        readback = _write_bit(
            client,
            reject_db,
            reject_byte,
            reject_bit,
            True,
        )

        if pulse_ms > 0:
            time.sleep(pulse_ms / 1000.0)
            _write_bit(
                client,
                reject_db,
                reject_byte,
                reject_bit,
                False,
            )

        return {
            "sent": readback is True,
            "display": "REJECT Sent" if readback is True else "REJECT Write Failed",
            "detail": f"REJECT readback={readback}",
        }

    except Exception as e:
        return {
            "sent": False,
            "display": "PLC Send Failed",
            "detail": str(e),
        }

    finally:
        try:
            if client is not None:
                client.disconnect()
        except Exception:
            pass