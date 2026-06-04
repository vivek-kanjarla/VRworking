#!/usr/bin/env bash
# linux_setup.sh — one-time Linux environment setup for Quest 3 → FR5 VR teleop.
#
# Run once per machine after cloning the repo:
#   chmod +x linux_setup.sh
#   ./linux_setup.sh
#
# What this script does:
#   1. Install Intel RealSense SDK + udev rules  (D405 wrist camera)
#   2. Install ADB                               (Quest 3 oculus_reader transport)
#   3. Install Python dependencies
#   4. Verify network interface for FR5          (192.168.58.x)
#   5. Check pynput display access               (keyboard E-stop / record)

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

info()  { echo -e "${BOLD}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}   $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}   $*"; }

echo ""
echo -e "${BOLD}=== Quest 3 → FR5 VR Teleop — Linux Setup ===${RESET}"
echo ""

# ── 1. Intel RealSense SDK ────────────────────────────────────────────────────

info "Checking for Intel RealSense SDK (librealsense2)..."
if dpkg -l librealsense2 &>/dev/null; then
    ok "librealsense2 already installed."
else
    warn "librealsense2 not found — installing from Intel's apt repo..."

    CODENAME=$(lsb_release -cs 2>/dev/null || echo "")
    if [ -z "$CODENAME" ]; then
        fail "Could not detect Ubuntu codename. Install librealsense2 manually:"
        fail "  https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md"
    else
        sudo mkdir -p /etc/apt/keyrings
        curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
            | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
        echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
https://librealsense.intel.com/Debian/apt-repo ${CODENAME} main" \
            | sudo tee /etc/apt/sources.list.d/librealsense.list > /dev/null
        sudo apt-get update -qq
        sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
        ok "librealsense2 installed."
    fi
fi

# udev rules so D405 is accessible without root
RULES_FILE="/etc/udev/rules.d/99-realsense-libusb.rules"
if [ -f "$RULES_FILE" ]; then
    ok "RealSense udev rules already in place."
else
    info "Installing RealSense udev rules..."
    RULES_SRC=$(dpkg -L librealsense2 2>/dev/null | grep -m1 99-realsense || true)
    if [ -n "$RULES_SRC" ] && [ -f "$RULES_SRC" ]; then
        sudo cp "$RULES_SRC" "$RULES_FILE"
        sudo udevadm control --reload-rules && sudo udevadm trigger
        ok "RealSense udev rules installed."
    else
        warn "Could not find packaged udev rules — download manually:"
        warn "  wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules"
        warn "  sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/"
        warn "  sudo udevadm control --reload-rules && sudo udevadm trigger"
    fi
fi

# ── 2. ADB (for Quest 3 oculus_reader transport) ─────────────────────────────

info "Checking ADB (needed for Quest 3 oculus_reader mode)..."
if command -v adb &>/dev/null; then
    ok "ADB found: $(adb version | head -1)"
else
    warn "ADB not found — installing android-tools-adb..."
    sudo apt-get install -y android-tools-adb \
        && ok "ADB installed." \
        || warn "ADB install failed — install manually: sudo apt-get install android-tools-adb"
fi

# Add user to plugdev group so ADB can access USB devices without root
if groups "$USER" | grep -qw plugdev; then
    ok "Already in plugdev group (ADB USB access)."
else
    warn "Adding $USER to plugdev group for ADB USB access..."
    sudo usermod -aG plugdev "$USER"
    warn "Log out and back in for group change to take effect."
fi

# ── 3. Python dependencies ────────────────────────────────────────────────────

info "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt --quiet \
    && ok "Python packages installed." \
    || warn "Some packages failed — check output above."

echo ""
warn "fairino SDK: install from Fairino's distribution .whl (not on PyPI):"
warn "  pip install /path/to/fairino-*.whl"
echo ""

# ── 4. Network interface check ────────────────────────────────────────────────

FR5_SUBNET="192.168.58"
info "Checking for a network interface on the ${FR5_SUBNET}.x subnet (FR5)..."

IFACE=$(ip -4 addr show | awk '/inet '"${FR5_SUBNET//./\\.}"'/{print $NF}' | head -1)
if [ -n "$IFACE" ]; then
    IP_ADDR=$(ip -4 addr show dev "$IFACE" | awk '/inet /{print $2}')
    ok "Interface $IFACE has IP $IP_ADDR — FR5 subnet looks configured."
else
    warn "No interface on ${FR5_SUBNET}.x subnet found."
    warn "Configure the Ethernet port connected to FR5. Example (replace eth0 / .10):"
    warn "  sudo ip addr add 192.168.58.10/24 dev eth0"
    warn "  sudo ip link set eth0 up"
    warn "For persistent config: edit /etc/netplan/ or use nm-connection-editor."
fi

# ── 5. pynput / keyboard access ──────────────────────────────────────────────

info "Checking pynput display access (needed for Space/R/H keyboard controls)..."
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    warn "No DISPLAY or WAYLAND_DISPLAY detected."
    warn "pynput requires a display session for the keyboard listener."
    warn "If running over SSH, either:"
    warn "  ssh -X user@host          (X11 forwarding)"
    warn "  export DISPLAY=:0         (if a desktop is already running on the machine)"
    warn "Or run teleop_vr.py directly at the physical console."
else
    ok "Display set: ${DISPLAY:-${WAYLAND_DISPLAY}}"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}=== Setup complete. Manual steps remaining: ===${RESET}"
echo ""
echo "  1. Install fairino .whl:  pip install /path/to/fairino-*.whl"
echo "  2. If added to plugdev:   LOG OUT and back in."
echo "  3. If Ethernet not set:   see FR5 subnet instructions above."
echo "  4. Quest 3 oculus_reader: adb install oculus_reader.apk"
echo "                            adb forward tcp:5555 tcp:5555"
echo "  5. Verify hardware:       python check_hardware.py"
echo "                            python check_network.py"
echo ""
