resource "oci_identity_compartment" "this" {
  for_each       = var.compartments
  compartment_id = lookup(each.value, "parent_compartment_id", var.compartment_id)
  name           = each.key
  description    = lookup(each.value, "description", "Child compartment created by AgentMesh")
  enable_delete  = lookup(each.value, "enable_delete", true)
}

output "compartment_ids" {
  value = { for k, v in oci_identity_compartment.this : k => v.id }
}
