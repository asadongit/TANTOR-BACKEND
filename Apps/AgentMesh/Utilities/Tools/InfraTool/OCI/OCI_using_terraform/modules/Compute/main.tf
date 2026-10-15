resource "oci_core_instance" "this" {
  for_each            = var.instances
  compartment_id      = var.compartment_id
  availability_domain = each.value.availability_domain
  shape               = each.value.shape
  display_name        = each.key
  state               = lookup(each.value, "state", "RUNNING")

  create_vnic_details {
    subnet_id        = each.value.subnet_id
    assign_public_ip = lookup(each.value, "assign_public_ip", true)
    nsg_ids          = lookup(each.value, "nsg_ids", null)
  }

  source_details {
    source_id   = each.value.image_id
    source_type = "image"
  }

  dynamic "shape_config" {
    for_each = lookup(each.value, "ocpus", null) != null || lookup(each.value, "memory_in_gbs", null) != null ? [1] : []
    content {
      ocpus         = lookup(each.value, "ocpus", null)
      memory_in_gbs = lookup(each.value, "memory_in_gbs", null)
    }
  }

  metadata = {
    ssh_authorized_keys = lookup(each.value, "ssh_authorized_key", null)
  }
}

data "oci_core_instances" "resolved_vms" {
  for_each       = { for k, v in var.boot_volume_backups : k => v if lookup(v, "vm_name", null) != null && (lookup(v, "boot_volume_id", null) == null || lookup(v, "boot_volume_id", "") == "") }
  compartment_id = var.compartment_id
  display_name   = each.value.vm_name
}

data "oci_core_boot_volume_attachments" "resolved_attachments" {
  for_each            = { for k, v in var.boot_volume_backups : k => v if lookup(v, "vm_name", null) != null && (lookup(v, "boot_volume_id", null) == null || lookup(v, "boot_volume_id", "") == "") }
  compartment_id      = var.compartment_id
  availability_domain = data.oci_core_instances.resolved_vms[each.key].instances[0].availability_domain
  instance_id         = data.oci_core_instances.resolved_vms[each.key].instances[0].id
}

resource "oci_core_boot_volume_backup" "this" {
  for_each       = var.boot_volume_backups
  boot_volume_id = lookup(each.value, "boot_volume_id", null) != null && lookup(each.value, "boot_volume_id", "") != "" ? each.value.boot_volume_id : data.oci_core_boot_volume_attachments.resolved_attachments[each.key].boot_volume_attachments[0].boot_volume_id
  display_name   = each.key
  type           = lookup(each.value, "type", "INCREMENTAL")
  compartment_id = var.compartment_id
}

output "instance_details" {
  value = { for k, v in oci_core_instance.this : k => {
    id         = v.id
    public_ip  = v.public_ip
    private_ip = v.private_ip
    state      = v.state
  }}
}

output "boot_volume_backup_ids" {
  value = { for k, v in oci_core_boot_volume_backup.this : k => v.id }
}
