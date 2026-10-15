variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "nsgs" {
  type = map(object({
    vcn_id = string
  }))
  description = "Map of NSG configurations"
  default     = {}
}

variable "nsg_rules" {
  type = map(object({
    nsg_name         = string
    direction        = string
    protocol         = string
    source           = optional(string)
    source_type      = optional(string)
    destination      = optional(string)
    destination_type = optional(string)
    stateless        = optional(bool)
    tcp_options = optional(object({
      min = number
      max = number
    }))
    udp_options = optional(object({
      min = number
      max = number
    }))
  }))
  description = "Map of NSG rules"
  default     = {}
}



