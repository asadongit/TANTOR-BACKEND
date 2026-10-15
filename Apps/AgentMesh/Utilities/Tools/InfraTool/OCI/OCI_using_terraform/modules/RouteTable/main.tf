resource "oci_core_route_table" "this" {
  for_each       = var.route_tables
  compartment_id = var.compartment_id
  vcn_id         = each.value.vcn_id
  display_name   = each.key

  dynamic "route_rules" {
    for_each = each.value.route_rules
    content {
      network_entity_id = route_rules.value.network_entity_id
      destination       = route_rules.value.destination
      destination_type  = lookup(route_rules.value, "destination_type", "CIDR_BLOCK")
    }
  }
}

output "route_table_ids" {
  value = { for k, v in oci_core_route_table.this : k => v.id }
}
