resource "oci_core_security_list" "this" {
  for_each       = var.security_lists
  compartment_id = var.compartment_id
  vcn_id         = each.value.vcn_id
  display_name   = each.key

  dynamic "egress_security_rules" {
    for_each = each.value.egress_rules
    content {
      destination      = egress_security_rules.value.destination
      destination_type = lookup(egress_security_rules.value, "destination_type", "CIDR_BLOCK")
      protocol         = egress_security_rules.value.protocol
      stateless        = lookup(egress_security_rules.value, "stateless", false)

      dynamic "tcp_options" {
        for_each = egress_security_rules.value.tcp_options != null ? [egress_security_rules.value.tcp_options] : []
        content {
          max = tcp_options.value.max
          min = tcp_options.value.min
        }
      }

      dynamic "udp_options" {
        for_each = egress_security_rules.value.udp_options != null ? [egress_security_rules.value.udp_options] : []
        content {
          max = udp_options.value.max
          min = udp_options.value.min
        }
      }
    }
  }

  dynamic "ingress_security_rules" {
    for_each = each.value.ingress_rules
    content {
      source      = ingress_security_rules.value.source
      source_type = lookup(ingress_security_rules.value, "source_type", "CIDR_BLOCK")
      protocol         = ingress_security_rules.value.protocol
      stateless        = lookup(ingress_security_rules.value, "stateless", false)

      dynamic "tcp_options" {
        for_each = ingress_security_rules.value.tcp_options != null ? [ingress_security_rules.value.tcp_options] : []
        content {
          max = tcp_options.value.max
          min = tcp_options.value.min
        }
      }

      dynamic "udp_options" {
        for_each = ingress_security_rules.value.udp_options != null ? [ingress_security_rules.value.udp_options] : []
        content {
          max = udp_options.value.max
          min = udp_options.value.min
        }
      }
    }
  }
}

output "security_list_ids" {
  value = { for k, v in oci_core_security_list.this : k => v.id }
}
