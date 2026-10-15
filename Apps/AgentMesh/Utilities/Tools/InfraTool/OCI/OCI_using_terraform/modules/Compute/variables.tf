variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "instances" {
  type = map(object({
    availability_domain = string
    shape               = string
    subnet_id           = string
    image_id            = string
    assign_public_ip    = optional(bool)
    nsg_ids             = optional(list(string))
    ssh_authorized_key  = optional(string)
    ocpus               = optional(number)
    memory_in_gbs       = optional(number)
    state               = optional(string)
  }))
  description = "Map of VM configurations"
  default     = {}
}

variable "boot_volume_backups" {
  type = map(object({
    boot_volume_id = optional(string)
    vm_name        = optional(string)
    type           = optional(string)
  }))
  description = "Map of Boot Volume backups"
  default     = {}
}
