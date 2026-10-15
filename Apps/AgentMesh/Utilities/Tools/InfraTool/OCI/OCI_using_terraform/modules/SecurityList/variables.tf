variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "security_lists" {
  type = map(object({
    vcn_id = string
    ingress_rules = optional(list(object({
      protocol    = string
      source      = string
      source_type = optional(string)
      stateless   = optional(bool)
      tcp_options = optional(object({
        min = number
        max = number
      }))
      udp_options = optional(object({
        min = number
        max = number
      }))
    })), [])
    egress_rules = optional(list(object({
      protocol         = string
      destination      = string
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
    })), [])
  }))
  description = "Map of Security List configurations"
  default     = {}
}
