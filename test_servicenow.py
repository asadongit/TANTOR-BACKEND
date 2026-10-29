#!/usr/bin/env python3
"""
test_servicenow.py - Standalone ServiceNow Integration Test Suite

Tests:
  Check D: Wrong credentials fail safely and cleanly (never treated as APPROVED)
  Check A: Ticket created programmatically, initial status is PENDING
  Check B: Human approves ticket in ServiceNow UI, status is APPROVED
  Check C: Human rejects a second ticket in ServiceNow UI, status is REJECTED

Usage:
  python3 test_servicenow.py
"""

import sys
import os
import time

# Ensure current directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from servicenow_client import (
    create_change_ticket,
    get_approval_status,
    get_assignment_group_id,
    _get_config
)

def print_separator(title=""):
    print("\n" + "=" * 65)
    if title:
        print(f"  {title}")
        print("=" * 65)

def main():
    print_separator("SERVICENOW VAPT INTEGRATION TEST SUITE")

    try:
        instance, user, _ = _get_config()
        print(f"Instance: {instance}")
        print(f"User:     {user}")
    except Exception as e:
        print(f"[FAIL] Configuration Error: {e}")
        sys.exit(1)

    # -------------------------------------------------------------
    # 0. Check Assignment Group Caching
    # -------------------------------------------------------------
    print_separator("0. ASSIGNMENT GROUP LOOKUP")
    group_id = get_assignment_group_id("Software")
    if group_id:
        print(f"[PASS] Retrieved and cached 'Software' group sys_id: {group_id}")
    else:
        print("[FAIL] Could not resolve sys_id for 'Software' group.")

    # -------------------------------------------------------------
    # Check D: Wrong credentials / safety fail-closed check
    # -------------------------------------------------------------
    print_separator("CHECK D: WRONG PASSWORD SECURITY / FAIL-CLOSED TEST")
    print("Testing get_approval_status with invalid credentials...")
    fake_auth = (user, "INCORRECT_PASSWORD_TEST_12345")
    status_with_wrong_pass = get_approval_status("any_sys_id_test", auth_override=fake_auth)
    
    if status_with_wrong_pass == "PENDING":
        print(f"[PASS] Check D: Wrong password cleanly returned '{status_with_wrong_pass}' (NEVER 'APPROVED').")
    else:
        print(f"[FAIL] Check D: Expected 'PENDING' on auth failure, but got '{status_with_wrong_pass}'.")
        sys.exit(1)

    # -------------------------------------------------------------
    # Check A: Create Ticket -> Status PENDING
    # -------------------------------------------------------------
    print_separator("CHECK A: TICKET CREATION & INITIAL PENDING STATUS")
    print("Creating test change ticket for approval flow...")
    number_a, sys_id_a, err_a = create_change_ticket(
        title="Check A/B Approval Verification",
        details="Automated test ticket for VAPT ServiceNow integration suite.",
        assignment_group_name="Software",
        auto_request_approval=True
    )

    if err_a or not sys_id_a:
        print(f"[FAIL] Check A: Ticket creation failed: {err_a}")
        sys.exit(1)

    print(f"Created Change Ticket: {number_a} (sys_id: {sys_id_a})")

    # Give ServiceNow 1 second to apply business rules
    time.sleep(1)
    status_a = get_approval_status(sys_id_a)
    print(f"Current Approval Status: {status_a}")

    if status_a == "PENDING":
        print(f"[PASS] Check A: Ticket {number_a} status is PENDING as expected.")
    else:
        print(f"[FAIL] Check A: Expected initial status PENDING, got: {status_a}")

    # -------------------------------------------------------------
    # Check B: Human Approves in UI -> Status APPROVED
    # -------------------------------------------------------------
    print_separator("CHECK B: MANUAL UI APPROVAL TEST")
    print(f"""
>>> MANUAL UI STEPS REQUIRED FOR TICKET {number_a}:
  1. Open ServiceNow PDI in your browser:
     {instance}
  2. Navigate to: 'Self-Service' -> 'My Approvals'
     (or search for {number_a} in Change -> Open)
  3. Open the approval record for {number_a} and click 'Approve'.
""")
    input(">>> Once you have approved the ticket in ServiceNow UI, press [ENTER] to verify: ")

    status_b = get_approval_status(sys_id_a)
    print(f"Re-checking status for {number_a}: {status_b}")
    if status_b == "APPROVED":
        print(f"[PASS] Check B: Ticket {number_a} successfully detected as APPROVED!")
    else:
        print(f"[FAIL] Check B: Expected 'APPROVED', but got '{status_b}'. (If you haven't clicked Approve yet, please approve and run again.)")

    # -------------------------------------------------------------
    # Check C: Create 2nd Ticket & Human Rejects in UI -> Status REJECTED
    # -------------------------------------------------------------
    print_separator("CHECK C: MANUAL UI REJECTION TEST")
    print("Creating second test change ticket for rejection flow...")
    number_c, sys_id_c, err_c = create_change_ticket(
        title="Check C Rejection Verification",
        details="Automated test ticket to verify human rejection handling.",
        assignment_group_name="Software",
        auto_request_approval=True
    )

    if err_c or not sys_id_c:
        print(f"[FAIL] Check C: Ticket creation failed: {err_c}")
        sys.exit(1)

    print(f"Created Change Ticket for Rejection: {number_c} (sys_id: {sys_id_c})")
    print(f"""
>>> MANUAL UI STEPS REQUIRED FOR TICKET {number_c}:
  1. Open ServiceNow PDI in your browser:
     {instance}
  2. Navigate to: 'Self-Service' -> 'My Approvals'
  3. Open the approval record for {number_c} and click 'Reject'.
""")
    input(">>> Once you have rejected the ticket in ServiceNow UI, press [ENTER] to verify: ")

    status_c = get_approval_status(sys_id_c)
    print(f"Re-checking status for {number_c}: {status_c}")
    if status_c == "REJECTED":
        print(f"[PASS] Check C: Ticket {number_c} successfully detected as REJECTED!")
    else:
        print(f"[FAIL] Check C: Expected 'REJECTED', but got '{status_c}'.")

    print_separator("ALL SERVICENOW CHECKS COMPLETE")

if __name__ == "__main__":
    main()
