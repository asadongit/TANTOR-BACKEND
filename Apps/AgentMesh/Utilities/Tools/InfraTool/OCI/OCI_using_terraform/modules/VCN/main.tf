resource "oci_core_vcn" "this" {
  for_each       = var.vcns
  compartment_id = var.compartment_id
  cidr_block     = each.value.cidr_block
  display_name   = each.key
  dns_label      = lookup(each.value, "dns_label", null)
}

output "vcn_ids" {
  value = { for k, v in oci_core_vcn.this : k => v.id }
}
