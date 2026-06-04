"""
check_network.py — verify network connectivity to the FR5 controller.
Run with: ~/teleop_env/bin/python3 ~/teleop/check_network.py
"""

import subprocess
import socket
import sys

FR5_IP   = "192.168.58.2"
FR5_PORT = 20003
TIMEOUT  = 3


def check_ping():
    print(f"[1/2] Pinging {FR5_IP} (3 packets)...", flush=True)
    result = subprocess.run(
        ["ping", "-c", "3", "-W", "2", FR5_IP],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        # Extract packet loss line for concise output
        for line in result.stdout.splitlines():
            if "packet loss" in line:
                print(f"      {line.strip()}")
        print(f"  PASS  ping {FR5_IP}")
        return True
    else:
        print(f"  FAIL  ping {FR5_IP} — no response")
        print("         Check: Is the FR5 controller powered on?")
        print("                Is your PC on the same subnet (192.168.58.x)?")
        print("                Run: ip addr show  — to confirm your IP")
        return False


def check_tcp():
    print(f"[2/2] TCP connect to {FR5_IP}:{FR5_PORT} (timeout {TIMEOUT}s)...", flush=True)
    try:
        sock = socket.create_connection((FR5_IP, FR5_PORT), timeout=TIMEOUT)
        sock.close()
        print(f"  PASS  TCP {FR5_IP}:{FR5_PORT} — Fairino RPC port reachable")
        return True
    except socket.timeout:
        print(f"  FAIL  TCP {FR5_IP}:{FR5_PORT} — connection timed out")
        print("         Check: Is the robot controller running?")
        print("                Firewall blocking port 20003?")
        return False
    except ConnectionRefusedError:
        print(f"  FAIL  TCP {FR5_IP}:{FR5_PORT} — connection refused")
        print("         The host is reachable but the RPC service is not listening.")
        print("         Check controller software / robot mode.")
        return False
    except OSError as exc:
        print(f"  FAIL  TCP {FR5_IP}:{FR5_PORT} — {exc}")
        return False


if __name__ == "__main__":
    print("=" * 55)
    print("  FR5 Network Connectivity Check")
    print("=" * 55)
    ok_ping = check_ping()
    ok_tcp  = check_tcp()
    print("=" * 55)
    if ok_ping and ok_tcp:
        print("  All checks PASSED — network looks good.")
        sys.exit(0)
    else:
        print("  One or more checks FAILED — see messages above.")
        sys.exit(1)
