# Compartments Query (Read-Only)
data "oci_identity_compartments" "all_compartments" {
  compartment_id            = var.tenancy_ocid
  access_level              = "ACCESSIBLE"
  compartment_id_in_subtree = true
}

# VCN Module (Provisioning)
module "vcn" {
  source         = "./modules/VCN"
  compartment_id = var.compartment_id
  vcns           = var.vcns
}

# Subnet Module (Provisioning)
module "subnet" {
  source         = "./modules/Subnet"
  compartment_id = var.compartment_id
  subnets        = var.subnets
}

# IGW Module (Provisioning)
module "igw" {
  source         = "./modules/IGW"
  compartment_id = var.compartment_id
  igws           = var.igws
}

# Security List Module (Provisioning)
# Security List Module (Provisioning)
module "security_list" {
  source         = "./modules/SecurityList"
  compartment_id = var.compartment_id
  security_lists = var.security_lists
}

# NSG Module (Provisioning)
module "nsg" {
  source                = "./modules/NSG"
  compartment_id        = var.compartment_id
  nsgs                  = var.nsgs
  nsg_rules             = var.nsg_rules
}

# List all VCNs in the compartment
data "oci_core_vcns" "all_vcns" {
  compartment_id = var.compartment_id
}

# List all Subnets in the compartment/VCN
data "oci_core_subnets" "all_subnets" {
  compartment_id = var.compartment_id
  vcn_id         = var.vcn_id != "" ? var.vcn_id : null
}

# List all Internet Gateways in the compartment/VCN
data "oci_core_internet_gateways" "all_igws" {
  compartment_id = var.compartment_id
  vcn_id         = var.vcn_id != "" ? var.vcn_id : null
}

# List all Security Lists in the compartment/VCN
data "oci_core_security_lists" "all_security_lists" {
  compartment_id = var.compartment_id
  vcn_id         = var.vcn_id != "" ? var.vcn_id : null
}

# List all Network Security Groups in the compartment/VCN
data "oci_core_network_security_groups" "all_nsgs" {
  compartment_id = var.compartment_id
  vcn_id         = var.vcn_id != "" ? var.vcn_id : null
}

# Compute Module (Provisioning)
module "compute" {
  source              = "./modules/Compute"
  compartment_id      = var.compartment_id
  instances           = var.instances
  boot_volume_backups = var.boot_volume_backups
}

# List all Compute Instances in the compartment
data "oci_core_instances" "all_instances" {
  compartment_id = var.compartment_id
}

# List all shapes in the compartment
data "oci_core_shapes" "all_shapes" {
  compartment_id = var.compartment_id
  image_id       = var.shape_filter_image_id != "" ? var.shape_filter_image_id : null
}

# List all images in the compartment
data "oci_core_images" "all_images" {
  compartment_id   = var.compartment_id
  operating_system = var.image_filter_operating_system != "" ? var.image_filter_operating_system : null
}

# DRG Module (Provisioning)
module "drg" {
  source          = "./modules/DRG"
  compartment_id  = var.compartment_id
  drgs            = var.drgs
  drg_attachments = var.drg_attachments
}

# List all DRGs in the compartment
data "oci_core_drgs" "all_drgs" {
  compartment_id = var.compartment_id
}

# List all DRG attachments in the compartment
data "oci_core_drg_attachments" "all_drg_attachments" {
  compartment_id = var.compartment_id
  drg_id         = var.drg_filter_id != "" ? var.drg_filter_id : null
}

# Route Table Module (Provisioning)
module "route_table" {
  source         = "./modules/RouteTable"
  compartment_id = var.compartment_id
  route_tables   = var.route_tables
}

# List all Route Tables in the compartment
data "oci_core_route_tables" "all_route_tables" {
  compartment_id = var.compartment_id
  vcn_id         = var.vcn_id != "" ? var.vcn_id : null
}

# Block Volume Module (Provisioning)
module "block_volume" {
  source             = "./modules/BlockVolume"
  compartment_id     = var.compartment_id
  volumes            = var.volumes
  volume_attachments = var.volume_attachments
  volume_backups     = var.volume_backups
}

# List all Block Volumes in the compartment
data "oci_core_volumes" "all_volumes" {
  compartment_id = var.compartment_id
}

# List all Volume attachments in the compartment
data "oci_core_volume_attachments" "all_volume_attachments" {
  compartment_id = var.compartment_id
  instance_id    = var.volume_filter_instance_id != "" ? var.volume_filter_instance_id : null
}

# List Availability Domains in the tenancy
data "oci_identity_availability_domains" "all_ads" {
  compartment_id = var.tenancy_ocid
}

# Compartment Module (Provisioning)
module "compartment" {
  source         = "./modules/Compartment"
  compartment_id = var.compartment_id
  compartments   = var.compartments
}








