output "compartments_json" {
  value       = data.oci_identity_compartments.all_compartments.compartments
  description = "List of all compartments in the tenancy"
}

output "vcn_ids" {
  value       = module.vcn.vcn_ids
  description = "Map of created VCN names to their OCIDs"
}

output "all_vcns" {
  value       = data.oci_core_vcns.all_vcns.virtual_networks
  description = "List of all VCNs in the compartment"
}

output "subnet_ids" {
  value       = module.subnet.subnet_ids
  description = "Map of created subnet names to their OCIDs"
}

output "all_subnets" {
  value       = data.oci_core_subnets.all_subnets.subnets
  description = "List of all subnets"
}

output "igw_ids" {
  value       = module.igw.igw_ids
  description = "Map of created IGW names to their OCIDs"
}

output "all_igws" {
  value       = data.oci_core_internet_gateways.all_igws.gateways
  description = "List of all Internet Gateways"
}

output "security_list_ids" {
  value       = module.security_list.security_list_ids
  description = "Map of created Security List names to their OCIDs"
}

output "all_security_lists" {
  value       = data.oci_core_security_lists.all_security_lists.security_lists
  description = "List of all Security Lists"
}

output "nsg_ids" {
  value       = module.nsg.nsg_ids
  description = "Map of created NSG names to their OCIDs"
}

output "all_nsgs" {
  value       = data.oci_core_network_security_groups.all_nsgs.network_security_groups
  description = "List of all NSGs"
}

output "instance_details" {
  value       = module.compute.instance_details
  description = "Map of created instances to details"
}

output "all_instances" {
  value       = data.oci_core_instances.all_instances.instances
  description = "List of all instances"
}

output "all_shapes" {
  value       = data.oci_core_shapes.all_shapes.shapes
  description = "List of all shapes"
}

output "all_images" {
  value       = data.oci_core_images.all_images.images
  description = "List of all images"
}

output "drg_ids" {
  value       = module.drg.drg_ids
  description = "Map of created DRG names to their OCIDs"
}

output "drg_attachment_ids" {
  value       = module.drg.drg_attachment_ids
  description = "Map of created DRG attachments to their OCIDs"
}

output "all_drgs" {
  value       = data.oci_core_drgs.all_drgs.drgs
  description = "List of all DRGs"
}

output "all_drg_attachments" {
  value       = data.oci_core_drg_attachments.all_drg_attachments.drg_attachments
  description = "List of all DRG attachments"
}

output "route_table_ids" {
  value       = module.route_table.route_table_ids
  description = "Map of created Route Table names to their OCIDs"
}

output "all_route_tables" {
  value       = data.oci_core_route_tables.all_route_tables.route_tables
  description = "List of all Route Tables"
}

output "volume_ids" {
  value       = module.block_volume.volume_ids
  description = "Map of created Volume names to their OCIDs"
}

output "volume_attachment_ids" {
  value       = module.block_volume.volume_attachment_ids
  description = "Map of created Volume attachments to their OCIDs"
}

output "all_volumes" {
  value       = data.oci_core_volumes.all_volumes.volumes
  description = "List of all Block Volumes"
}

output "all_volume_attachments" {
  value       = data.oci_core_volume_attachments.all_volume_attachments.volume_attachments
  description = "List of all Volume attachments"
}

output "all_ads" {
  value       = data.oci_identity_availability_domains.all_ads.availability_domains
  description = "List of all availability domains"
}

output "volume_backup_ids" {
  value       = module.block_volume.volume_backup_ids
  description = "Map of created Block Volume backups to their OCIDs"
}

output "boot_volume_backup_ids" {
  value       = module.compute.boot_volume_backup_ids
  description = "Map of created Boot Volume backups to their OCIDs"
}

output "compartment_ids" {
  value       = module.compartment.compartment_ids
  description = "Map of created compartment names to their OCIDs"
}









