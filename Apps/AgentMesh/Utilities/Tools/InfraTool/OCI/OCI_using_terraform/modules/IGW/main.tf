resource "oci_core_internet_gateway" "this" {
  for_each       = var.igws
  compartment_id = var.compartment_id
  vcn_id         = each.value.vcn_id
  display_name   = each.key
  enabled        = lookup(each.value, "enabled", true)
}

output "igw_ids" {
  value = { for k, v in oci_core_internet_gateway.this : k => v.id }
}
