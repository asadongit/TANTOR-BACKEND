resource "oci_core_network_security_group" "this" {
  for_each       = var.nsgs
  compartment_id = var.compartment_id
  vcn_id         = each.value.vcn_id
  display_name   = each.key
}

resource "oci_core_network_security_group_security_rule" "rules" {
  for_each                  = var.nsg_rules
  network_security_group_id = oci_core_network_security_group.this[each.value.nsg_name].id
  direction                 = each.value.direction
  protocol                  = each.value.protocol
  source                    = lookup(each.value, "source", null)
  source_type               = lookup(each.value, "source_type", null)
  destination               = lookup(each.value, "destination", null)
  destination_type          = lookup(each.value, "destination_type", null)
  stateless                 = lookup(each.value, "stateless", false)

  dynamic "tcp_options" {
    for_each = each.value.tcp_options != null ? [each.value.tcp_options] : []
    content {
      destination_port_range {
        max = tcp_options.value.max
        min = tcp_options.value.min
      }
    }
  }

  dynamic "udp_options" {
    for_each = each.value.udp_options != null ? [each.value.udp_options] : []
    content {
      destination_port_range {
        max = udp_options.value.max
        min = udp_options.value.min
      }
    }
  }
}


output "nsg_ids" {
  value = { for k, v in oci_core_network_security_group.this : k => v.id }
}
