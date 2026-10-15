#!/bin/bash
set -e

echo "================================================="
echo " LVM CLEANUP AND DISK DETACH PREP SCRIPT"
echo "================================================="

# Detect if arguments are passed (non-interactive mode)
if [ -n "$1" ]; then
    PV_DISK=$1
    VG_NAME=$2
    LV_NAME=$3
    MOUNT_POINT=$4
    LV_PATH="/dev/$VG_NAME/$LV_NAME"
    CONFIRM="YES"
    echo "Running in Non-Interactive mode (Arguments passed)"
else
    # Interactive mode
    echo -e "\n[Available Disks]"
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
    echo -e "\n[Filesystem Usage]"
    df -h
    echo -e "\n[Physical Volumes]"
    pvs || true
    echo -e "\n[Volume Groups]"
    vgs || true
    echo -e "\n[Logical Volumes]"
    lvs || true
    echo -e "\n[Current Mounts]"
    mount | grep '^/dev' || true
    echo ""
    echo "================================================="
    echo "Enter LVM Details"
    echo "================================================="
    read -p "Enter Mount Point (e.g. /data): " MOUNT_POINT
    read -p "Enter LV Path (e.g. /dev/vg_data/lv_data): " LV_PATH
    read -p "Enter VG Name (e.g. vg_data): " VG_NAME
    read -p "Enter PV Disk (e.g. /dev/sdb): " PV_DISK
    echo ""
    echo "================================================="
    echo "Confirmation"
    echo "================================================="
    echo "Mount Point : $MOUNT_POINT"
    echo "LV Path : $LV_PATH"
    echo "VG Name : $VG_NAME"
    echo "PV Disk : $PV_DISK"
    echo ""
    read -p "Type YES to permanently delete this LVM configuration: " CONFIRM
fi

if [ "$CONFIRM" != "YES" ]; then
    echo "Operation cancelled."
    exit 1
fi

echo ""
echo "================================================="
echo "Starting Cleanup"
echo "================================================="

# Unmount safely
if [ -n "$MOUNT_POINT" ] && (mountpoint -q "$MOUNT_POINT" || grep -qs " $MOUNT_POINT " /proc/mounts); then
    echo "Unmounting filesystem..."
    sudo umount -f "$MOUNT_POINT" || sudo umount -l "$MOUNT_POINT"
else
    echo "Mount point $MOUNT_POINT is not mounted."
fi

# Clean up /etc/fstab
if [ -f /etc/fstab ]; then
    echo "Cleaning up /etc/fstab..."
    if [ -n "$VG_NAME" ] && [ -n "$LV_NAME" ]; then
        sudo sed -i "\|/dev/$VG_NAME/$LV_NAME|d" /etc/fstab
        sudo sed -i "\|/dev/mapper/${VG_NAME}-${LV_NAME}|d" /etc/fstab
    fi
    if [ -n "$MOUNT_POINT" ]; then
        sudo sed -i "\| $MOUNT_POINT |d" /etc/fstab
    fi
fi

# Remove Logical Volume
if [ -n "$VG_NAME" ] && [ -n "$LV_NAME" ]; then
    LV_PATH="/dev/$VG_NAME/$LV_NAME"
    MAPPER_PATH="/dev/mapper/${VG_NAME}-${LV_NAME}"
    if sudo lvs "$LV_PATH" &>/dev/null || [ -L "$LV_PATH" ] || [ -L "$MAPPER_PATH" ]; then
        echo "Removing Logical Volume..."
        sudo lvremove -f -y "$LV_PATH" || sudo lvremove -f -y "$MAPPER_PATH"
    else
        echo "Logical Volume $LV_NAME does not exist."
    fi
fi

# Remove Volume Group
if [ -n "$VG_NAME" ]; then
    if sudo vgs "$VG_NAME" &>/dev/null; then
        echo "Removing Volume Group..."
        sudo vgremove -f -y "$VG_NAME"
    else
        echo "Volume Group $VG_NAME does not exist."
    fi
fi

# Remove Physical Volume
if [ -n "$PV_DISK" ]; then
    # Resolve partitioned PV name if applicable
    if [[ "$PV_DISK" =~ [0-9]$ ]]; then
        PARTITION="${PV_DISK}p1"
    else
        PARTITION="${PV_DISK}1"
    fi
    
    for PV in "$PARTITION" "$PV_DISK"; do
        if [ -b "$PV" ] && sudo pvs "$PV" &>/dev/null; then
            echo "Removing Physical Volume: $PV"
            sudo pvremove -f -y "$PV"
        fi
    done
fi

# Wipe partition table and disk signatures
if [ -n "$PV_DISK" ] && [ -b "$PV_DISK" ]; then
    echo "Wiping Partition and Disk Signatures on $PV_DISK..."
    if sudo parted -s "$PV_DISK" print &>/dev/null; then
        sudo parted -s "$PV_DISK" rm 1 2>/dev/null || true
    fi
    sudo wipefs -a "$PV_DISK"
fi

echo ""
echo "================================================="
echo "Verification"
echo "================================================="
echo -e "\n[Available Disks]"
sudo lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
echo -e "\n[Physical Volumes]"
sudo pvs || true
echo -e "\n[Volume Groups]"
sudo vgs || true
echo -e "\n[Logical Volumes]"
sudo lvs || true
echo -e "\n[Filesystem Usage]"
df -h
echo ""
echo "================================================="
echo "LVM Cleanup Completed Successfully"
echo "================================================="
echo ""
echo "Next Step:"
echo "Detach the corresponding OCI Block Volume from:"
echo "OCI Console -> Compute -> Instance -> Attached Block Volumes -> Detach"
echo ""