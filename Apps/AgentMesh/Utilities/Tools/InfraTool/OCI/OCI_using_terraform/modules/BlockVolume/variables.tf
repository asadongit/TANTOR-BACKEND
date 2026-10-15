variable "compartment_id" {
  type        = string
  description = "Compartment OCID"
}

variable "volumes" {
  type = map(object({
    availability_domain = string
    size_in_gbs         = optional(number, 50)
  }))
  description = "Map of Block Volume configurations"
  default     = {}
}

variable "volume_attachments" {
  type = map(object({
    attachment_type = string
    instance_id     = string
    volume_id       = string
  }))
  description = "Map of Block Volume attachments"
  default     = {}
}

variable "volume_backups" {
  type = map(object({
    volume_id = string
    type      = optional(string)
  }))
  description = "Map of Block Volume backups"
  default     = {}
}
