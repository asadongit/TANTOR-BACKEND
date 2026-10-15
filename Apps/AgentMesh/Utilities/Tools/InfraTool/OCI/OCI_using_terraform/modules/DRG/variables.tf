variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "drgs" {
  type        = map(object({}))
  description = "Map of DRG configurations"
  default     = {}
}

variable "drg_attachments" {
  type = map(object({
    drg_id = string
    vcn_id = string
  }))
  description = "Map of DRG attachments"
  default     = {}
}
