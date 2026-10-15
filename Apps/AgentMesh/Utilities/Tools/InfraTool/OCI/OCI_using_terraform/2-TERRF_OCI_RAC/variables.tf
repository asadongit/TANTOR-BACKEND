variable "tenancy_ocid" {
  type = string
}

variable "user_ocid" {
  type = string
}

variable "fingerprint" {
  type = string
}

variable "private_key_path" {
  type = string
}

variable "region" {
  type = string
}

variable "compartment_id" {
  type = string
}

variable "vcn_name" {
  type = string
}

variable "vcn_cidr" {
  type = string
}

variable "public_subnet_name" {
  type = string
}

variable "public_subnet_cidr" {
  type = string
}

variable "private_subnet_name" {
  type = string
}

variable "private_subnet_cidr" {
  type = string
}

variable "instances" {

  type = map(object({

    availability_domain = string

    shape = string

    ocpus = number

    memory_in_gbs = number

    assign_public_ip = bool

    ssh_public_key_path = string

    image_ocid = string

    boot_volume_size_in_gbs = number

    private_ip = optional(string)

    hostname_label = string
  }))
}

variable "vip_ips" {

  type = map(object({

    instance_name = string

    ip_address = optional(string)
  }))
}

variable "scan_ips" {

  type = map(object({

    instance_name = string

    ip_address = optional(string)
  }))
}

variable "local_block_volumes" {

  type = map(object({

    instance_name = string

    size_in_gbs = number
  }))
}

variable "shared_block_volume" {

  type = object({

    display_name = string

    size_in_gbs = number
  })
}

variable "network_config" {

  type = object({

    public_route_table = object({

      display_name = string

      route_rules = list(object({

        destination = string

        destination_type = string
      }))
    })

    public_subnet = object({

      cidr_block = string

      display_name = string

      dns_label = string

      prohibit_public_ip_on_vnic = bool
    })

    private_subnet = object({

      cidr_block = string

      display_name = string

      dns_label = string

      prohibit_public_ip_on_vnic = bool
    })

    public_security_list = object({

      display_name = string

      ingress_security_rules = list(object({

        protocol = string

        source = string

        tcp_options = optional(object({

          min = number

          max = number
        }))
      }))

      egress_security_rules = list(object({

        protocol = string

        destination = string
      }))
    })

    private_security_list = object({

      display_name = string

      ingress_security_rules = list(object({

        protocol = string

        source = string

        tcp_options = optional(object({

          min = number

          max = number
        }))
      }))

      egress_security_rules = list(object({

        protocol = string

        destination = string
      }))
    })
  })
}

