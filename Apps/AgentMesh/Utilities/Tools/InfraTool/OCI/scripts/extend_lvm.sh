#!/bin/bash


echo "========================================"
echo "Extend Existing LVM"
echo "========================================"


# Check if arguments are provided (Non-Interactive Mode)
if [ $# -gt 0 ]; then
    PARTITION=$1
    VG_NAME=${2:-"vg_agentic"}
    LV_NAME=${3:-"lv_agentic"}
if [ $# -gt 0 ]; then
    PARTITION=$1
    VG_NAME=${2:-"vg_agentic"}
    LV_NAME=${3:-"lv_agentic"}
    
    echo "Running in Non-Interactive Mode:"
    echo "Partition:   $PARTITION"
    echo "VG Name:     $VG_NAME"
    echo "LV Name:     $LV_NAME"
    echo "Partition:   $PARTITION"
    echo "VG Name:     $VG_NAME"
    echo "LV Name:     $LV_NAME"
else
    # Fallback to Interactive Mode if no arguments are passed
    read -p "Enter Existing Disk Partition (Example: /dev/sdb1): " PARTITION
    read -p "Enter VG Name: " VG_NAME
    read -p "Enter LV Name: " LV_NAME
fi


# Validate Partition
if [ ! -b "$PARTITION" ]; then
    echo "Partition '$PARTITION' does not exist."
if [ ! -b "$PARTITION" ]; then
    echo "Partition '$PARTITION' does not exist."
    exit 1
fi


echo "----------------------------------------"
echo "Rescanning Disk"
echo "----------------------------------------"


# Dynamically extract partition number and disk name
# E.g. /dev/sdb1 -> disk: /dev/sdb, part: 1. /dev/nvme0n1p2 -> disk: /dev/nvme0n1, part: 2.
PART_NUM=$(echo "$PARTITION" | grep -o -E '[0-9]+$')
DISK=$(echo "$PARTITION" | sed -E 's/p?[0-9]+$//')
PART_NUM=$(echo "$PARTITION" | grep -o -E '[0-9]+$')
DISK=$(echo "$PARTITION" | sed -E 's/p?[0-9]+$//')

if [ -z "$PART_NUM" ] || [ -z "$DISK" ]; then
    echo "No partition number detected. Treating '$PARTITION' as a raw disk."
    DISK=$PARTITION
if [ -z "$PART_NUM" ] || [ -z "$DISK" ]; then
    echo "No partition number detected. Treating '$PARTITION' as a raw disk."
    DISK=$PARTITION
    PART_NUM=""
fi

echo "Disk: $DISK"
if [ -n "$PART_NUM" ]; then
    echo "Partition Number: $PART_NUM"
echo "Disk: $DISK"
if [ -n "$PART_NUM" ]; then
    echo "Partition Number: $PART_NUM"
fi

echo 1 | sudo tee /sys/class/block/$(basename $DISK)/device/rescan 2>/dev/null || true


echo 1 | sudo tee /sys/class/block/$(basename $DISK)/device/rescan 2>/dev/null || true

# Wait for disk to settle
sudo udevadm settle 2>/dev/null || sleep 5

if [ -n "$PART_NUM" ]; then

if [ -n "$PART_NUM" ]; then
    echo "----------------------------------------"
    echo "Growing Partition"
    echo "----------------------------------------"
    sudo growpart $DISK $PART_NUM
    sudo growpart $DISK $PART_NUM
else
    echo "----------------------------------------"
    echo "Skipping growpart for raw disk"
    echo "----------------------------------------"
fi


echo "----------------------------------------"
echo "Resizing Physical Volume"
echo "----------------------------------------"

sudo pvresize $PARTITION


sudo pvresize $PARTITION

echo "----------------------------------------"
echo "Extending Logical Volume"
echo "----------------------------------------"

sudo lvextend -y -l +100%FREE /dev/$VG_NAME/$LV_NAME


sudo lvextend -y -l +100%FREE /dev/$VG_NAME/$LV_NAME

echo "----------------------------------------"
echo "Getting Mount Point and Filesystem Type"
echo "----------------------------------------"

MOUNT_POINT=$(findmnt -n -o TARGET /dev/$VG_NAME/$LV_NAME)
FSTYPE=$(findmnt -n -o FSTYPE /dev/$VG_NAME/$LV_NAME)

echo "Mount Point:     $MOUNT_POINT"
echo "Filesystem Type: $FSTYPE"

echo "----------------------------------------"
echo "Growing Filesystem"
echo "----------------------------------------"

if [ -z "$MOUNT_POINT" ]; then

if [ -z "$MOUNT_POINT" ]; then
    echo "Error: Device is not mounted. Resizing the filesystem directly on the volume..."
    if [[ "$FSTYPE" == *"ext"* ]]; then
        sudo resize2fs /dev/$VG_NAME/$LV_NAME
    if [[ "$FSTYPE" == *"ext"* ]]; then
        sudo resize2fs /dev/$VG_NAME/$LV_NAME
    else
        echo "Please mount the partition to resize XFS."
    fi
elif [ "$FSTYPE" = "xfs" ]; then
    echo "Resizing XFS filesystem at $MOUNT_POINT..."
    sudo xfs_growfs $MOUNT_POINT
elif [[ "$FSTYPE" == *"ext"* ]]; then
    echo "Resizing Ext filesystem at /dev/$VG_NAME/$LV_NAME..."
    sudo resize2fs /dev/$VG_NAME/$LV_NAME
elif [ "$FSTYPE" = "xfs" ]; then
    echo "Resizing XFS filesystem at $MOUNT_POINT..."
    sudo xfs_growfs $MOUNT_POINT
elif [[ "$FSTYPE" == *"ext"* ]]; then
    echo "Resizing Ext filesystem at /dev/$VG_NAME/$LV_NAME..."
    sudo resize2fs /dev/$VG_NAME/$LV_NAME
else
    echo "Unknown filesystem type ($FSTYPE). Trying standard resize utilities..."
    sudo xfs_growfs $MOUNT_POINT 2>/dev/null || sudo resize2fs /dev/$VG_NAME/$LV_NAME
    echo "Unknown filesystem type ($FSTYPE). Trying standard resize utilities..."
    sudo xfs_growfs $MOUNT_POINT 2>/dev/null || sudo resize2fs /dev/$VG_NAME/$LV_NAME
fi


echo "----------------------------------------"
echo "LVM Extension Completed"
echo "----------------------------------------"


df -h
lsblk