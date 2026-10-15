resource "oci_core_subnet" "this" {
  for_each          = var.subnets
  compartment_id    = var.compartment_id
  vcn_id            = each.value.vcn_id
  cidr_block        = each.value.cidr_block
  display_name      = each.key
  dns_label         = lookup(each.value, "dns_label", null)
  security_list_ids = lookup(each.value, "security_list_ids", null)
  route_table_id    = lookup(each.value, "route_table_id", null)
}

output "subnet_ids" {
  value = { for k, v in oci_core_subnet.this : k => v.id }
}
