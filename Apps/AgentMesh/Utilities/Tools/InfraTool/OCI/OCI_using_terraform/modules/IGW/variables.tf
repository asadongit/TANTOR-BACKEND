variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "igws" {
  type = map(object({
    vcn_id  = string
    enabled = optional(bool)
  }))
  description = "Map of IGW configurations"
  default     = {}
}
