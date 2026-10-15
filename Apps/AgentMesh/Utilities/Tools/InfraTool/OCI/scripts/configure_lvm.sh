#!/bin/bash

echo "========================================"
echo "New Disk LVM Configuration (XFS)"
echo "========================================"

# Check if arguments are provided (Non-Interactive Mode)
if [ $# -gt 0 ]; then
    DISK=$1
    VG_NAME=${2:-"vg_agentic"}
    LV_NAME=${3:-"lv_agentic"}
    MOUNT_POINT=${4:-"/agentic"}
    LV_SIZE=${5:-"100%FREE"}
    
    if [ -n "$6" ]; then
        PARTITION=$6
    else
        # Automatically detect partition name (e.g., /dev/sdb -> /dev/sdb1, /dev/nvme0n1 -> /dev/nvme0n1p1)
        if [[ "$DISK" =~ [0-9]$ ]]; then
            PARTITION="${DISK}p1"
        else
            PARTITION="${DISK}1"
        fi
    fi
    
    echo "Running in Non-Interactive Mode:"
    echo "Disk:        $DISK"
    echo "Partition:   $PARTITION"
    echo "VG Name:     $VG_NAME"
    echo "LV Name:     $LV_NAME"
    echo "LV Size:     $LV_SIZE"
    echo "Mount Point: $MOUNT_POINT"
else
    # Fallback to Interactive Mode if no arguments are passed
    read -p "Enter Disk Name (Example: /dev/sdb): " DISK
    read -p "Enter Partition Name (Example: /dev/sdb1): " PARTITION
    read -p "Enter VG Name: " VG_NAME
    read -p "Enter LV Name: " LV_NAME
    read -p "Enter LV Size (Example: 10G or 100%FREE): " LV_SIZE
    read -p "Enter Mount Point (Example: /data): " MOUNT_POINT
fi

# Validate Disk
if [ ! -b "$DISK" ]; then
    echo "Disk '$DISK' does not exist or is not a block device."
    exit 1
fi

# Install packages
if ! command -v pvcreate &> /dev/null; then
    if command -v yum &> /dev/null; then
        sudo yum install -y lvm2 xfsprogs parted
    else
        sudo apt-get update -y
        sudo apt-get install -y lvm2 xfsprogs parted
    fi
fi

echo "----------------------------------------"
echo "Creating Partition"
echo "----------------------------------------"

sudo parted -s $DISK mklabel gpt
sudo parted -s $DISK mkpart primary 0% 100%

# Wait for OS / udev to register the partition
sudo udevadm settle 2>/dev/null || sleep 3

echo "----------------------------------------"
echo "Creating Physical Volume"
echo "----------------------------------------"

sudo pvcreate -y $PARTITION

echo "----------------------------------------"
echo "Creating Volume Group"
echo "----------------------------------------"

sudo vgcreate -y $VG_NAME $PARTITION

echo "----------------------------------------"
echo "Creating Logical Volume"
echo "----------------------------------------"

if [[ "$LV_SIZE" == *"FREE"* ]] || [[ "$LV_SIZE" == *"free"* ]]; then
    sudo lvcreate -y -l $LV_SIZE -n $LV_NAME $VG_NAME
else
    sudo lvcreate -y -L $LV_SIZE -n $LV_NAME $VG_NAME
fi

echo "----------------------------------------"
echo "Formatting XFS Filesystem"
echo "----------------------------------------"

sudo mkfs.xfs -f /dev/$VG_NAME/$LV_NAME

echo "----------------------------------------"
echo "Creating Mount Point"
echo "----------------------------------------"

sudo mkdir -p $MOUNT_POINT

echo "----------------------------------------"
echo "Mounting Filesystem"
echo "----------------------------------------"

sudo mount /dev/$VG_NAME/$LV_NAME $MOUNT_POINT

echo "----------------------------------------"
echo "Persisting in /etc/fstab"
echo "----------------------------------------"

UUID=$(sudo blkid -s UUID -o value /dev/$VG_NAME/$LV_NAME)

if [ -n "$UUID" ]; then
    echo "UUID=$UUID $MOUNT_POINT xfs defaults,_netdev,nofail 0 0" | sudo tee -a /etc/fstab
else
    echo "/dev/$VG_NAME/$LV_NAME $MOUNT_POINT xfs defaults,_netdev,nofail 0 0" | sudo tee -a /etc/fstab
fi

echo "----------------------------------------"
echo "LVM Configuration Completed"
echo "----------------------------------------"

df -h
lsblk