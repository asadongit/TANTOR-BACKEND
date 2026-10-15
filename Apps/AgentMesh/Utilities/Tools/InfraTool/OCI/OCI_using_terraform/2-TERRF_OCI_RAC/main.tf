# --- compute ---
resource "oci_core_instance" "this" {

  for_each = var.instances

  compartment_id = var.compartment_id

  availability_domain = each.value.availability_domain

  display_name = each.key

  shape = each.value.shape

  dynamic "shape_config" {
    for_each = length(regexall(".*\\.Flex", each.value.shape)) > 0 ? [1] : []
    content {
      ocpus = each.value.ocpus
      memory_in_gbs = each.value.memory_in_gbs
    }
  }

  create_vnic_details {

    subnet_id = oci_core_subnet.public_subnet.id

    assign_public_ip = each.value.assign_public_ip

    private_ip = lookup(each.value, "private_ip", null)

    hostname_label = each.value.hostname_label

    skip_source_dest_check = true
  }

  source_details {

    source_type = "image"

    source_id = each.value.image_ocid

    boot_volume_size_in_gbs = each.value.boot_volume_size_in_gbs
  }

  metadata = {

    ssh_authorized_keys = file(each.value.ssh_public_key_path)
  }
}

data "oci_core_vnic_attachments" "primary_vnic_attachments" {

  for_each = oci_core_instance.this

  compartment_id = var.compartment_id

  instance_id = each.value.id
}

data "oci_core_vnic" "primary_vnics" {

  for_each = oci_core_instance.this

  vnic_id = data.oci_core_vnic_attachments.primary_vnic_attachments[each.key].vnic_attachments[0].vnic_id
}

# --- network ---
resource "oci_core_vcn" "this" {

  compartment_id = var.compartment_id

  cidr_blocks = [var.vcn_cidr]

  display_name = var.vcn_name

  dns_label = "racvcn"
}

resource "oci_core_internet_gateway" "this" {

  compartment_id = var.compartment_id

  vcn_id = oci_core_vcn.this.id

  display_name = "${var.vcn_name}-igw"

  enabled = true
}

resource "oci_core_route_table" "public_rt" {

  compartment_id = var.compartment_id

  vcn_id = oci_core_vcn.this.id

  display_name = var.network_config.public_route_table.display_name

  dynamic "route_rules" {

    for_each = var.network_config.public_route_table.route_rules

    content {

      destination = route_rules.value.destination

      destination_type = route_rules.value.destination_type

      network_entity_id = oci_core_internet_gateway.this.id
    }
  }
}

resource "oci_core_subnet" "public_subnet" {

  compartment_id = var.compartment_id

  vcn_id = oci_core_vcn.this.id

  cidr_block = var.network_config.public_subnet.cidr_block

  display_name = var.network_config.public_subnet.display_name

  dns_label = var.network_config.public_subnet.dns_label

  prohibit_public_ip_on_vnic = var.network_config.public_subnet.prohibit_public_ip_on_vnic

  security_list_ids = [oci_core_security_list.public_sl.id]

  route_table_id = oci_core_route_table.public_rt.id

}

resource "oci_core_subnet" "private_subnet" {

  compartment_id = var.compartment_id

  vcn_id = oci_core_vcn.this.id

  cidr_block = var.network_config.private_subnet.cidr_block

  display_name = var.network_config.private_subnet.display_name

  dns_label = var.network_config.private_subnet.dns_label

  prohibit_public_ip_on_vnic = var.network_config.private_subnet.prohibit_public_ip_on_vnic

  security_list_ids = [oci_core_security_list.private_sl.id]
}

resource "oci_core_security_list" "public_sl" {

  compartment_id = var.compartment_id

  vcn_id = oci_core_vcn.this.id

  display_name = var.network_config.public_security_list.display_name

  dynamic "ingress_security_rules" {

    for_each = var.network_config.public_security_list.ingress_security_rules

    content {

      protocol = ingress_security_rules.value.protocol

      source = ingress_security_rules.value.source

      dynamic "tcp_options" {

        for_each = lookup(ingress_security_rules.value, "tcp_options", null) != null ? [1] : []

        content {

          min = ingress_security_rules.value.tcp_options.min

          max = ingress_security_rules.value.tcp_options.max
        }
      }
    }
  }

  dynamic "egress_security_rules" {

    for_each = var.network_config.public_security_list.egress_security_rules

    content {

      protocol = egress_security_rules.value.protocol

      destination = egress_security_rules.value.destination
    }
  }
}

resource "oci_core_security_list" "private_sl" {

  compartment_id = var.compartment_id

  vcn_id = oci_core_vcn.this.id

  display_name = var.network_config.private_security_list.display_name

  dynamic "ingress_security_rules" {

    for_each = var.network_config.private_security_list.ingress_security_rules

    content {

      protocol = ingress_security_rules.value.protocol

      source = ingress_security_rules.value.source

      dynamic "tcp_options" {

        for_each = lookup(ingress_security_rules.value, "tcp_options", null) != null ? [1] : []

        content {

          min = ingress_security_rules.value.tcp_options.min

          max = ingress_security_rules.value.tcp_options.max
        }
      }
    }
  }

  dynamic "egress_security_rules" {

    for_each = var.network_config.private_security_list.egress_security_rules

    content {

      protocol = egress_security_rules.value.protocol

      destination = egress_security_rules.value.destination
    }
  }
}


# --- private-ips ---
resource "oci_core_private_ip" "vip_ips" {

  for_each = var.vip_ips

  vnic_id = data.oci_core_vnic_attachments.primary_vnic_attachments[each.value.instance_name].vnic_attachments[0].vnic_id

  ip_address = lookup(each.value, "ip_address", null)

  display_name = each.key
  hostname_label = each.key
}


resource "oci_core_private_ip" "scan_ips" {

  for_each = var.scan_ips

  vnic_id = data.oci_core_vnic_attachments.primary_vnic_attachments[each.value.instance_name].vnic_attachments[0].vnic_id

  ip_address = lookup(each.value, "ip_address", null)

  display_name = each.key
  hostname_label = each.key
}

# --- storage ---
resource "oci_core_volume" "local_volumes" {

  for_each = var.local_block_volumes

  availability_domain = values(var.instances)[0].availability_domain

  compartment_id = var.compartment_id

  display_name = each.key

  size_in_gbs = each.value.size_in_gbs
}

resource "oci_core_volume_attachment" "local_volume_attachments" {

  for_each = var.local_block_volumes

  attachment_type = "paravirtualized"

  instance_id = oci_core_instance.this[each.value.instance_name].id

  volume_id = oci_core_volume.local_volumes[each.key].id

  display_name = "${each.key}-attachment"

  is_read_only = false
}

resource "oci_core_volume" "shared_asm_volume" {

  availability_domain = values(var.instances)[0].availability_domain

  compartment_id = var.compartment_id

  display_name = var.shared_block_volume.display_name

  size_in_gbs = var.shared_block_volume.size_in_gbs

}

resource "oci_core_volume_attachment" "shared_volume_attachments" {

  for_each = oci_core_instance.this

  attachment_type = "paravirtualized"

  instance_id = each.value.id

  volume_id = oci_core_volume.shared_asm_volume.id

  display_name = "${each.key}-shared-asm-attachment"

  is_read_only = false

  is_shareable = true
}

# --- vnic ---
resource "oci_core_vnic_attachment" "secondary_vnic" {

  for_each = oci_core_instance.this

  instance_id = each.value.id

  create_vnic_details {

    subnet_id = oci_core_subnet.private_subnet.id

    assign_public_ip = false

    skip_source_dest_check = true

    display_name = "${each.key}-private-vnic"
  }

  display_name = "${each.key}-private-vnic"


}


provider "oci" {

  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}
terraform {

  required_version = ">= 1.5.0"

  required_providers {

    oci = {
      source  = "oracle/oci"
      version = ">= 5.0.0"
    }
  }
}