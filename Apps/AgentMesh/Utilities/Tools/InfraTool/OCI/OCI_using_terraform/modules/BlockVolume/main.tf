resource "oci_core_volume" "this" {
  for_each            = var.volumes
  compartment_id      = var.compartment_id
  availability_domain = each.value.availability_domain
  display_name        = each.key
  size_in_gbs         = each.value.size_in_gbs
}

resource "oci_core_volume_attachment" "this" {
  for_each        = var.volume_attachments
  attachment_type = each.value.attachment_type
  instance_id     = each.value.instance_id
  volume_id       = each.value.volume_id
  display_name    = each.key
}

resource "oci_core_volume_backup" "this" {
  for_each       = var.volume_backups
  volume_id      = each.value.volume_id
  display_name   = each.key
  type           = lookup(each.value, "type", "INCREMENTAL")
  compartment_id = var.compartment_id
}

output "volume_ids" {
  value = { for k, v in oci_core_volume.this : k => v.id }
}

output "volume_attachment_ids" {
  value = { for k, v in oci_core_volume_attachment.this : k => v.id }
}

output "volume_backup_ids" {
  value = { for k, v in oci_core_volume_backup.this : k => v.id }
}
