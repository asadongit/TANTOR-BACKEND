variable "tenancy_ocid" {
  type        = string
  description = "OCI Tenancy OCID"
}

variable "user_ocid" {
  type        = string
  description = "OCI User OCID"
}

variable "fingerprint" {
  type        = string
  description = "OCI API Key Fingerprint"
}

variable "private_key_path" {
  type        = string
  description = "Path to the temporary private key file"
}

variable "region" {
  type        = string
  description = "OCI Region"
}

variable "compartment_id" {
  type        = string
  description = "The OCID of the target compartment"
}

# Network Variables (Defaulted to empty to allow other tools to run without declaring them)
variable "vcns" {
  type = map(object({
    cidr_block = string
    dns_label  = optional(string)
  }))
  description = "Map of VCN configurations"
  default     = {}
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

variable "vcn_id" {
  type        = string
  description = "VCN OCID for listing subnets"
  default     = ""
}

variable "igws" {
  type = map(object({
    vcn_id  = string
    enabled = optional(bool)
  }))
  description = "Map of IGW configurations"
  default     = {}
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

variable "image_filter_operating_system" {
  type        = string
  description = "Operating system to filter images by"
  default     = ""
}

variable "shape_filter_image_id" {
  type        = string
  description = "Image OCID to filter shapes by"
  default     = ""
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

variable "drg_filter_id" {
  type        = string
  description = "DRG OCID to filter attachments by"
  default     = ""
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

variable "volumes" {
  type = map(object({
    availability_domain = string
    size_in_gbs         = optional(number)
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

variable "volume_filter_instance_id" {
  type        = string
  description = "Compute Instance OCID to filter volume attachments by"
  default     = ""
}

variable "volume_backups" {
  type = map(object({
    volume_id = string
    type      = optional(string)
  }))
  description = "Map of Block Volume backups"
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

variable "compartments" {
  type = map(object({
    parent_compartment_id = optional(string)
    description           = optional(string)
    enable_delete         = optional(bool, true)
  }))
  description = "Map of OCI compartment configurations"
  default     = {}
}











