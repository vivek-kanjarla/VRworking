#!/usr/bin/env bash
# START_WIFI.sh — FR5 VR Sim Launcher for Quest 3 connected via WiFi (no USB ADB)
# Usage: ./START_WIFI.sh [sim|validate|execute]
#
# sim      = Quest 3 controller drives FR5 in RViz only (no hardware)
# validate = sim validates each command, then sends to real FR5
# execute  = direct VR→FR5 (no sim)

MODE=${1:-sim}
LAPTOP_IP="192.168.0.6"   # update if your WiFi IP changes (run: ip addr show wlp8s0)

echo "=== FR5 VR Launcher — WiFi mode (mode=$MODE) ==="
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  On the Quest 3 headset:                            │"
echo "  │  1. Open Meta Browser                               │"
echo "  │  2. Go to: https://${LAPTOP_IP}:8012                 │"
echo "  │  3. Accept cert warning → tap TAP ANYWHERE          │"
echo "  │  4. Squeeze right trigger → move wrist              │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

# Kill any leftover processes from a previous run
echo "[1/3] Cleaning up old processes..."
pkill -f "sim_bridge\|vr_sim_launch\|robot_state_publisher\|rviz2" 2>/dev/null || true
sleep 1
fuser -k 8012/tcp 8013/tcp 2>/dev/null || true
sleep 1

# Launch
echo "[2/3] Starting sim stack (mode=$MODE)..."

cd "$(dirname "$0")"

PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/opt/ros/humble/lib \
PYTHONUNBUFFERED=1 \
DISPLAY="${DISPLAY:-:1}" \
~/teleop_env/bin/python3 vr_sim_launch.py --mode "$MODE"
