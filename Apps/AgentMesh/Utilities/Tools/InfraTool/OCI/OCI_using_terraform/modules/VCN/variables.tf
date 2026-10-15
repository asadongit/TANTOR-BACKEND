variable "compartment_id" {
  type        = string
  description = "OCI Compartment OCID"
}

variable "vcns" {
  type = map(object({
    cidr_block = string
    dns_label  = optional(string)
  }))
  description = "Map of VCN configurations"
}
