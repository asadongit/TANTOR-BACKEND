resource "oci_core_drg" "this" {
  for_each       = var.drgs
  compartment_id = var.compartment_id
  display_name   = each.key
}

resource "oci_core_drg_attachment" "this" {
  for_each     = var.drg_attachments
  drg_id       = each.value.drg_id
  display_name = each.key

  network_details {
    id   = each.value.vcn_id
    type = "VCN"
  }
}

output "drg_ids" {
  value = { for k, v in oci_core_drg.this : k => v.id }
}

output "drg_attachment_ids" {
  value = { for k, v in oci_core_drg_attachment.this : k => v.id }
}
