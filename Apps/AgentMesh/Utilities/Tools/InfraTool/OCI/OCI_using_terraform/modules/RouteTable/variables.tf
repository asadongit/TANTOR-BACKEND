variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "route_tables" {
  type = map(object({
    vcn_id = string
    route_rules = optional(list(object({
      network_entity_id = string
      destination       = string
      destination_type  = optional(string)
    })), [])
  }))
  description = "Map of Route Table configurations"
  default     = {}
}
