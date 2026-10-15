variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "subnets" {
  type = map(object({
    vcn_id            = string
    cidr_block        = string
    dns_label         = optional(string)
    security_list_ids = optional(list(string))
    route_table_id    = optional(string)
  }))
  description = "Map of subnet configurations"
  default     = {}
}
