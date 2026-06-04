#!/usr/bin/env bash
# START_SIM.sh — one-command launch for FR5 VR simulation
# Usage: ./START_SIM.sh [sim|validate|execute]
#
# sim      = Quest 3 controller drives FR5 in RViz only (no hardware)
# validate = sim validates each command, then sends to real FR5
# execute  = direct VR→FR5 (no sim)

MODE=${1:-sim}

echo "=== FR5 VR Sim Launcher (mode=$MODE) ==="

# 1. Reinstall APK if Quest is connected (safe to run every time)
if adb get-state 2>/dev/null | grep -q "device"; then
    echo "[1/4] Quest 3 connected — setting up ADB tunnels..."
    adb reverse tcp:8012 tcp:8012 2>/dev/null
    adb reverse tcp:8013 tcp:8013 2>/dev/null
else
    echo "[1/4] Quest 3 not connected via USB — connect and retry"
    exit 1
fi

# 2. Kill any leftover processes
echo "[2/4] Cleaning up old processes..."
pkill -f "sim_bridge\|vr_sim_launch\|robot_state_publisher\|rviz2" 2>/dev/null
sleep 1
fuser -k 8012/tcp 8013/tcp 2>/dev/null
sleep 1

# 3. Launch
echo "[3/4] Starting sim stack (mode=$MODE)..."
echo ""
echo "  ┌─────────────────────────────────────────────┐"
echo "  │  On the Quest 3 headset:                    │"
echo "  │  1. Open Meta Browser                       │"
echo "  │  2. Go to: https://192.168.0.10:8012        │"
echo "  │  3. Accept cert → tap TAP ANYWHERE          │"
echo "  │  4. Squeeze right trigger → move wrist      │"
echo "  └─────────────────────────────────────────────┘"
echo ""

cd "$(dirname "$0")"

PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/opt/ros/humble/lib \
PYTHONUNBUFFERED=1 \
~/teleop_env/bin/python3 vr_sim_launch.py --mode "$MODE"
