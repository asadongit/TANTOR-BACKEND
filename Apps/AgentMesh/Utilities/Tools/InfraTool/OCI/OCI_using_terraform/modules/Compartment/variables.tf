variable "compartment_id" {
  type        = string
  description = "Default parent compartment OCID (usually tenancy_ocid)"
}

variable "compartments" {
  type = map(object({
    parent_compartment_id = optional(string)
    description           = optional(string)
    enable_delete         = optional(bool, true)
  }))
  default = {}
}
