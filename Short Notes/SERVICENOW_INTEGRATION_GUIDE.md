# Complete ServiceNow Integration Guide: From Scratch to Production

---

## 1. Why Are We Doing This? (The Big Picture)

### The Problem in Enterprise Security
Your tool is an **Agentic AI VAPT (Vulnerability Assessment and Penetration Testing) remediation system**. 
When a scan finds critical security bugs (like outdated OpenSSL or vulnerable Linux kernels), the AI can automatically generate bash commands (`apt-get upgrade`, `systemctl restart`, etc.) and connect via SSH to patch the server.

**However, in real companies, an AI is NEVER allowed to touch production servers without human permission.**
If an AI restarts a server during peak business hours, it could crash a bank or an e-commerce website.

### The Solution: ServiceNow ITIL Change Management
Enterprises use **ServiceNow** to track every change to their infrastructure. Before any server is touched:
1. A **Change Request Ticket** must be created.
2. A human manager or security lead reviews the plan and clicks **Approve** in ServiceNow.
3. Only when ServiceNow says **"Approved"**, the AI is allowed to execute the SSH script.
4. After patching finishes, the AI writes the results back into ServiceNow for IT compliance and auditing.

---

## 2. What Was Done Previously (The Initial Setup with Claude)

Before our conversation, you and Claude performed the initial setup on ServiceNow:

1. **ServiceNow Developer Instance (PDI)**:
   - You registered a free Personal Developer Instance: `https://dev432009.service-now.com`.
   - This instance is your private sandbox containing real ServiceNow tables, workflows, and UI.

2. **The Integration User (`vapt_integration`)**:
   - Instead of using the super-administrator account for API calls, you created a dedicated technical user: `vapt_integration`.
   - You assigned it the **`itil`** role, which gives it permission to create, read, and update Change Requests via API, but not full admin rights.

3. **The Assignment Group (`Software`)**:
   - In ServiceNow, every change ticket must be assigned to a responsible team (an Assignment Group).
   - In your ServiceNow instance, there is a built-in group named **"Software"** containing 5 members (David Loo, Beth Anglin, Fred Luddy, ITIL User, Don Goodliffe).

4. **The Credentials in `.env`**:
   - To keep passwords secure, credentials are saved in `/home/rahul/Documents/AgentMesh/.env`:
     ```env
     SNOW_INSTANCE=https://dev432009.service-now.com
     SNOW_USER=vapt_integration
     SNOW_PASS=your_vapt_integration_password
     ```
   - `.env` is added to `.gitignore` so your password is never pushed to GitHub.

5. **The Prototype Script (`snow_test.py`)**:
   - A basic 38-line script was written that tested `requests.post()` to create a ticket and `requests.get()` to check it.

---

## 3. How ServiceNow Actually Works Internally

ServiceNow is built on relational database tables accessed via a standard REST API:

### A. The Change Request Table (`change_request`)
Every change ticket lives in the `change_request` table. Important fields:
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `number` | String | Human-readable ticket ID (e.g. `CHG0030012`) |
| `sys_id` | 32-char UUID | Unique database primary key (e.g. `c1159d2b2f88...`) used in API calls |
| `state` | Integer/Choice | Current lifecycle phase (`-5`: New, `-4`: Assess, `-3`: Authorize, `3`: Closed) |
| `approval` | Choice | The master approval flag (`not requested`, `requested`, `approved`, `rejected`) |
| `assignment_group` | Reference | Sys_id of the group responsible (e.g. `Software`) |
| `work_notes` | Journal/Text | Audit trail where the AI logs execution notes |

### B. The 2-Step Approval Process (Why you saw multiple approvers)
ServiceNow uses ITIL "Normal Change" rules. A Normal Change has **two distinct approval gates**:
1. **Gate 1: Technical Assessment (Assess Phase, State `-4`)**
   - The ticket is sent to the `Software` group members.
   - Any member (like `ITIL User`) reviews technical risk and clicks **Approve**.
2. **Gate 2: CAB Approval (Authorize Phase, State `-3`)**
   - Once the technical team approves, ServiceNow automatically forwards the ticket to the **CAB (Change Advisory Board)**: Christen Mitchell, Luke Wilson, etc.
   - When CAB clicks **Approve**, the entire ticket's master `approval` field flips from `requested` to **`approved`**.

---

## 4. What We Implemented Right Now (The New Python Code)

To make this rock-solid, production-ready, and fail-safe, we built two standalone files:

### File 1: `servicenow_client.py`
This is your clean, reusable ServiceNow library. It handles all network calls:

1. **`get_assignment_group_id("Software")`**:
   - Queries `GET /api/now/table/sys_user_group?name=Software`.
   - Caches the sys_id (`8a4dde73c6112278017a6a4baf547aa7`) in memory so it doesn't waste network calls querying it again.
2. **`create_change_ticket(title, description, ...)`**:
   - Creates the ticket in ServiceNow.
   - **Key Discovery**: By passing `state: "-4"` (Assess) and `assignment_group` during creation, ServiceNow **automatically requests approvals immediately**! Nobody needs to manually open the ticket and click "Request Approval".
3. **`get_approval_status(sys_id)`**:
   - Checks the ticket's `approval` field.
   - **Fail-Closed Safety**: If ServiceNow returns `"approved"`, it returns `"APPROVED"`. If `"rejected"`, it returns `"REJECTED"`. If there is a network error, timeout (15s), wrong password (401), or the ticket is still waiting, it returns `"PENDING"`. **It NEVER falsely returns "APPROVED".**
4. **`add_work_note(sys_id, text)`**:
   - Appends audit logs directly to the ticket activity log via `PATCH`.
5. **`close_ticket(sys_id, close_notes)`**:
   - Attempts to close the ticket. Captures the ServiceNow Business Rule gracefully if the ticket is still in review.

---

### File 2: `test_servicenow.py`
This is your interactive test suite. When you ran it, here is what it verified:

* **CHECK 0 (Group Lookup)**: Found group `Software` and cached it. `[PASS]`
* **CHECK D (Wrong Password)**: Intentionally gave a bad password. Caught HTTP 401 and returned `PENDING` without crashing. `[PASS]`
* **CHECK A (Ticket Creation)**: Created `CHG0030012` in state `Assess` (`-4`), status `PENDING`. `[PASS]`
* **CHECK B (Human Approval)**: You opened ServiceNow in your browser and clicked **Approve**. Python checked ServiceNow and detected:
  ```
  Re-checking status for CHG0030012: APPROVED
  [PASS] Check B: Ticket CHG0030012 successfully detected as APPROVED!
  ```
* **CHECK C (Human Rejection)**: Created `CHG0030013`. You clicked **Reject** in ServiceNow. Python detected:
  ```
  Re-checking status for CHG0030013: REJECTED
  [PASS] Check C: Ticket CHG0030013 successfully detected as REJECTED!
  ```

---

## 5. How We Implement It Into the Chat Flow Now

Here is how the user interacts with the AI in the chat:

```
[User Chat]                             [Agent Backend]                         [ServiceNow PDI]
     │                                         │                                       │
     │ 1. "I want to remediate OS critical"    │                                       │
     ├────────────────────────────────────────►│                                       │
     │                                         │ 2. Generates bash fix script          │
     │                                         │                                       │
     │                                         │ 3. create_change_ticket()             │
     │                                         ├──────────────────────────────────────►│ (Ticket Created: CHG0030015)
     │                                         │                                       │ (Approval: requested)
     │ 4. "Plan generated, ticket CHG0030015   │◄──────────────────────────────────────┤
     │     created. Once approved, type proceed"                                       │
     │◄────────────────────────────────────────┤                                       │
     │                                         │                                       │
     │                                         │                                       │
     │       [Human Manager opens ServiceNow UI and clicks Approve]                    │
     │                                                                                 │
     │                                         │                                       │
     │ 5. User types: "proceed"                │                                       │
     ├────────────────────────────────────────►│                                       │
     │                                         │ 6. get_approval_status(sys_id)        │
     │                                         ├──────────────────────────────────────►│
     │                                         │◄──────────────────────────────────────┤ Returns "approved"
     │                                         │                                       │
     │ 7. "Ticket approved! Enter machine IP"  │                                       │
     │◄────────────────────────────────────────┤                                       │
     │                                         │                                       │
     │ 8. User gives IP & Username             │                                       │
     ├────────────────────────────────────────►│                                       │
     │                                         │ 9. Runs execute_ssh_mitigation()      │
     │                                         │    (Patches Linux server via SSH)     │
     │                                         │                                       │
     │                                         │ 10. add_work_note()                   │
     │                                         ├──────────────────────────────────────►│ (Logs patch output to ticket)
     │ 11. "Patching complete! Results: ..."   │                                       │
     │◄────────────────────────────────────────┤                                       │
```

---

## 6. How It Works in the Future (Production & Enterprise Scaling)

When this moves to real corporate production, here is what changes:

1. **Authentication (OAuth 2.0 / SSO)**:
   - Instead of basic username/password in `.env`, production ServiceNow uses OAuth2 Client ID/Secret or mutual TLS certificates.
2. **Dynamic Routing**:
   - If the vulnerability is in **Database (Postgres/Oracle)** -> Ticket automatically routes to `DBA Group`.
   - If the vulnerability is in **Network/Firewall** -> Ticket routes to `Network Security Group`.
   - If the vulnerability is in **Linux/OS** -> Ticket routes to `Software / SysAdmin Group`.
3. **Change Windows**:
   - ServiceNow checks if the change is scheduled inside an approved maintenance window (e.g. Saturday 2:00 AM - 4:00 AM). The AI will only execute during that window.
4. **Closed-Loop Audit Compliance**:
   - Every SSH keystroke and terminal output is attached to the ServiceNow ticket.
   - If external auditors (SOC2, ISO 27001) inspect the company, they have 100% proof that:
     1. A vulnerability was found.
     2. A human approved the fix.
     3. The AI executed the exact approved script.
     4. Post-patch verification confirmed the vulnerability was closed.
