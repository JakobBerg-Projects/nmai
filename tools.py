"""High-level Tripletex tools with validation and multi-step workflow handling."""

import json
import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger("tools")

VAT_NUMBER_MAP = {25: "3", 12: "33", 15: "31", 0: "5", 6: "6"}
ELEVATED_ROLES = {
    "ALL_PRIVILEGES", "ACCOUNTANT", "INVOICING_MANAGER",
    "PERSONELL_MANAGER", "DEPARTMENT_LEADER", "AUDITOR",
}

TOOL_DEFINITIONS = [
    {
        "name": "create_employee",
        "description": (
            "Create an employee in Tripletex. Automatically handles: "
            "department requirement (retries with department if needed), "
            "email conflict (finds existing employee), "
            "role granting (upgrades userType to EXTENDED and grants entitlement template). "
            "Searches for existing employee first to avoid duplicates. "
            "Use for: opprett ansatt, create employee, Mitarbeiter erstellen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumberMobile": {"type": "string"},
                "dateOfBirth": {"type": "string", "description": "YYYY-MM-DD"},
                "employeeNumber": {"type": "string"},
                "nationalIdentityNumber": {"type": "string"},
                "startDate": {"type": "string", "description": "YYYY-MM-DD employment start date"},
                "userType": {
                    "type": "string",
                    "enum": ["STANDARD", "EXTENDED", "NO_ACCESS"],
                    "description": "Default: STANDARD. EXTENDED for admin/manager.",
                },
                "role": {
                    "type": "string",
                    "description": (
                        "Entitlement template to grant: ALL_PRIVILEGES, ACCOUNTANT, "
                        "INVOICING_MANAGER, PERSONELL_MANAGER, DEPARTMENT_LEADER, AUDITOR. "
                        "Auto-sets userType to EXTENDED."
                    ),
                },
                "department": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
                    "description": "Department reference. Use name to auto-match from prefetched data.",
                },
                "address": {
                    "type": "object",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
                "allowInformationRegistration": {
                    "type": "boolean",
                    "description": "Default: true. Allow login.",
                },
            },
            "required": ["firstName", "lastName"],
        },
    },
    {
        "name": "create_customer",
        "description": (
            "Create a customer with optional contact persons. "
            "Sets isCustomer:true automatically. "
            "Use for: opprett kunde, create customer, Kunde erstellen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumber": {"type": "string"},
                "organizationNumber": {"type": "string"},
                "customerNumber": {"type": "integer"},
                "isPrivateIndividual": {"type": "boolean"},
                "invoiceSendMethod": {
                    "type": "string",
                    "enum": ["EMAIL", "EHF", "EFAKTURA", "VIPPS", "PAPER", "MANUAL"],
                },
                "website": {"type": "string"},
                "description": {"type": "string"},
                "invoiceEmail": {"type": "string", "description": "Separate email for invoices"},
                "address": {
                    "type": "object",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
                "contacts": {
                    "type": "array",
                    "description": "Contact persons to add after creation",
                    "items": {
                        "type": "object",
                        "properties": {
                            "firstName": {"type": "string"},
                            "lastName": {"type": "string"},
                            "email": {"type": "string"},
                            "phoneNumberMobile": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "create_invoice_workflow",
        "description": (
            "Complete invoice workflow: customer → order (with lines) → invoice → optional payment. "
            "Creates customer if needed. Handles deliveryDate, vatType number format, "
            "price including/excluding VAT conversion. "
            "Use for: opprett faktura, create invoice, Rechnung erstellen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customerName": {"type": "string", "description": "Creates customer if not existing."},
                "customerId": {"type": "integer", "description": "Use existing customer ID."},
                "customerEmail": {"type": "string"},
                "customerPhone": {"type": "string"},
                "customerOrgNumber": {"type": "string"},
                "isPrivateIndividual": {"type": "boolean"},
                "invoiceSendMethod": {
                    "type": "string",
                    "enum": ["EMAIL", "EHF", "EFAKTURA", "VIPPS", "PAPER", "MANUAL"],
                    "description": "How to send invoice. Default: EMAIL",
                },
                "customerAddress": {
                    "type": "object",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
                "contacts": {
                    "type": "array",
                    "description": "Contact persons to add to the customer",
                    "items": {
                        "type": "object",
                        "properties": {
                            "firstName": {"type": "string"},
                            "lastName": {"type": "string"},
                            "email": {"type": "string"},
                            "phoneNumberMobile": {"type": "string"},
                        },
                    },
                },
                "orderLines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "count": {"type": "number", "description": "Quantity. Default: 1"},
                            "unitPriceExcludingVat": {"type": "number"},
                            "unitPriceIncludingVat": {
                                "type": "number",
                                "description": "If given, excl. is auto-calculated.",
                            },
                            "vatPercent": {"type": "number", "description": "25 (default), 12, 0"},
                            "productId": {"type": "integer"},
                            "productName": {"type": "string", "description": "Creates product if needed"},
                            "productNumber": {"type": "string"},
                        },
                        "required": ["description"],
                    },
                },
                "orderDate": {"type": "string", "description": "YYYY-MM-DD. Default: today"},
                "deliveryDate": {"type": "string", "description": "YYYY-MM-DD. Default: orderDate"},
                "invoiceDate": {"type": "string", "description": "YYYY-MM-DD. Default: today"},
                "dueDate": {
                    "type": "string",
                    "description": "YYYY-MM-DD. If set, uses POST /invoice with invoiceDueDate.",
                },
                "registerPayment": {"type": "boolean", "description": "Register full payment after invoicing."},
                "paymentDate": {"type": "string", "description": "YYYY-MM-DD. Default: today"},
                "comment": {"type": "string", "description": "Comment on the order"},
            },
            "required": ["orderLines"],
        },
    },
    {
        "name": "manage_travel_expense",
        "description": (
            "Create or delete travel expense (reiseregning) or employee expense (ansattutlegg). "
            "Creation: shell → costs (separate endpoint) → mileage → per diem. "
            "Include travelDetails for reiseregning (required for mileage/per diem). "
            "Without travelDetails → ansattutlegg. "
            "isDayTrip is auto-detected from dates if not specified. "
            "Use for: reiseregning, ansattutlegg, travel expense, Reisekosten."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "delete"], "description": "Default: create"},
                "employeeId": {"type": "integer"},
                "employeeName": {
                    "type": "string",
                    "description": "Full name. Auto-searches for employee ID.",
                },
                "travelExpenseId": {"type": "integer", "description": "For deletion."},
                "title": {"type": "string"},
                "travelDetails": {
                    "type": "object",
                    "description": "Include for reiseregning. Omit for ansattutlegg.",
                    "properties": {
                        "departureDate": {"type": "string"},
                        "returnDate": {"type": "string"},
                        "departureFrom": {"type": "string"},
                        "destination": {"type": "string"},
                        "isDayTrip": {"type": "boolean", "description": "Auto-detected from dates if omitted"},
                        "isForeignTravel": {"type": "boolean"},
                    },
                },
                "costs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "amount": {"type": "number"},
                            "category": {
                                "type": "string",
                                "description": "Match to cost category: Fly, Hotell, Taxi, Mat, Tog, etc.",
                            },
                            "description": {"type": "string"},
                            "vatType": {"type": "string", "description": "VAT type number, e.g. '3' for 25%"},
                        },
                    },
                },
                "mileage": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "km": {"type": "number"},
                        "departureLocation": {"type": "string"},
                        "destination": {"type": "string"},
                        "rateTypeId": {"type": "integer"},
                    },
                },
                "perDiem": {
                    "type": "object",
                    "properties": {
                        "rateCategoryName": {
                            "type": "string",
                            "description": "e.g. 'Dagsreise over 12 timer', 'Døgn med overnatting'",
                        },
                        "rateCategoryId": {"type": "integer"},
                        "location": {"type": "string"},
                        "overnightAccommodation": {
                            "type": "string",
                            "enum": ["NONE", "HOTEL", "OTHER"],
                        },
                        "startDate": {"type": "string"},
                        "endDate": {"type": "string"},
                    },
                },
            },
            "required": ["employeeId"],
        },
    },
    {
        "name": "create_project_workflow",
        "description": (
            "Create a project with automatic manager setup. "
            "Handles: EXTENDED upgrade, DEPARTMENT_LEADER entitlement, "
            "category lookup/creation, customer for external projects. "
            "Use for: opprett prosjekt, create project, Projekt erstellen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "projectManagerEmployeeId": {"type": "integer"},
                "projectManagerName": {
                    "type": "string",
                    "description": "Full name. Searches/creates employee.",
                },
                "customerName": {"type": "string"},
                "customerId": {"type": "integer"},
                "isInternal": {
                    "type": "boolean",
                    "description": "Default: true if no customer, false with customer.",
                },
                "categoryName": {"type": "string"},
                "startDate": {"type": "string"},
                "endDate": {"type": "string"},
                "fixedPrice": {"type": "number"},
                "description": {"type": "string"},
                "number": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "create_voucher",
        "description": (
            "Create a ledger voucher with balanced postings. "
            "Looks up account IDs from numbers. Validates sum = 0. "
            "Use for: bilag, voucher, bokføring, opening balance, åpningsbalanse, year-end closing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "description": {"type": "string"},
                "postings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "accountNumber": {
                                "type": "integer",
                                "description": "e.g. 1920, 3000, 5000",
                            },
                            "amount": {
                                "type": "number",
                                "description": "Positive=debit, negative=credit",
                            },
                            "description": {"type": "string"},
                            "customerId": {"type": "integer"},
                            "supplierId": {"type": "integer"},
                            "employeeId": {"type": "integer"},
                            "vatTypeNumber": {
                                "type": "string",
                                "description": "VAT type number, e.g. '3' for 25%. Omit for no VAT.",
                            },
                        },
                        "required": ["accountNumber", "amount"],
                    },
                },
                "useOpeningBalance": {
                    "type": "boolean",
                    "description": "Use opening balance endpoint instead of regular voucher",
                },
            },
            "required": ["date", "description", "postings"],
        },
    },
    {
        "name": "tripletex_api",
        "description": (
            "Raw Tripletex API call. FALLBACK for anything not covered by other tools. "
            "Use for: GET searches, PUT updates (include version!), DELETE, "
            "POST /supplier, POST /product, POST /supplierInvoice, POST /timesheet/entry, "
            "POST /bank/statement/:import, POST /bank/reconciliation/match, "
            "and any endpoint the high-level tools don't support. "
            "Common paths: /employee, /customer, /supplier, /product, /order, /invoice, "
            "/travelExpense, /project, /ledger/voucher, /ledger/account, /contact, "
            "/timesheet/entry, /bank, /bank/statement, /bank/reconciliation, "
            "/supplierInvoice, /department"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                "path": {
                    "type": "string",
                    "description": "API path e.g. '/employee', '/order/123/:invoice'",
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters",
                    "additionalProperties": True,
                },
                "json_body": {
                    "type": "object",
                    "description": "JSON body for POST/PUT",
                    "additionalProperties": True,
                },
            },
            "required": ["method", "path"],
        },
    },
]


class ToolHandler:
    """Executes high-level tools with validation, defaults, and error recovery."""

    def __init__(self, client: Any, ref_data: dict[str, list]):
        self.client = client
        self.ref = ref_data
        self.today = date.today().isoformat()
        self._account_cache: dict[int, int] = {}

    async def execute(self, tool_name: str, tool_input: dict) -> str:
        handler = getattr(self, f"_do_{tool_name}", None)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}", "success": False})
        try:
            result = await handler(tool_input)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return json.dumps({"error": str(e)[:500], "success": False})

    # ─── helpers ────────────────────────────────────────────────────────

    async def _api(self, method: str, path: str, params=None, body=None) -> dict:
        return await self.client.request(method, path, params=params, json_body=body)

    def _ok(self, r: dict) -> bool:
        sc = r.get("status_code")
        return isinstance(sc, int) and sc < 400

    def _id(self, r: dict) -> int | None:
        d = r.get("data", {})
        v = d.get("value", d) if isinstance(d, dict) else {}
        return v.get("id") if isinstance(v, dict) else None

    def _vals(self, r: dict) -> list:
        d = r.get("data", {})
        return d.get("values", []) if isinstance(d, dict) else []

    def _ref_id(self, key: str) -> int | None:
        items = self.ref.get(key, [])
        return items[0]["id"] if items else None

    def _match_ref(self, key: str, field: str, search: str) -> int | None:
        items = self.ref.get(key, [])
        if not items:
            return None
        if not search:
            return items[0]["id"]
        sl = search.lower().strip()
        for item in items:
            val = str(item.get(field, "")).lower()
            if sl == val:
                return item["id"]
        for item in items:
            val = str(item.get(field, "")).lower()
            if sl in val or val in sl:
                return item["id"]
        for word in sl.split():
            if len(word) < 2:
                continue
            for item in items:
                val = str(item.get(field, "")).lower()
                if word in val:
                    return item["id"]
        return items[0]["id"]

    def _find_employee_in_ref(self, first: str, last: str) -> dict | None:
        employees = self.ref.get("employees", [])
        fl, ll = first.lower(), last.lower()
        for emp in employees:
            ef = str(emp.get("firstName", "")).lower()
            el = str(emp.get("lastName", "")).lower()
            if fl == ef and ll == el:
                return emp
        for emp in employees:
            ef = str(emp.get("firstName", "")).lower()
            el = str(emp.get("lastName", "")).lower()
            if fl in ef and ll in el:
                return emp
        return None

    def _find_customer_in_ref(self, name: str) -> dict | None:
        customers = self.ref.get("customers", [])
        nl = name.lower()
        for c in customers:
            if nl == str(c.get("name", "")).lower():
                return c
        for c in customers:
            if nl in str(c.get("name", "")).lower():
                return c
        return None

    def _find_supplier_in_ref(self, name: str) -> dict | None:
        suppliers = self.ref.get("suppliers", [])
        nl = name.lower()
        for s in suppliers:
            if nl == str(s.get("name", "")).lower():
                return s
        for s in suppliers:
            if nl in str(s.get("name", "")).lower():
                return s
        return None

    async def _search_employee(self, first: str, last: str) -> dict | None:
        ref_match = self._find_employee_in_ref(first, last)
        if ref_match:
            return ref_match
        sr = await self._api("GET", "/employee", params={
            "firstName": first, "lastName": last,
            "fields": "id,firstName,lastName,userType,version,email",
        })
        vals = self._vals(sr)
        return vals[0] if vals else None

    async def _grant_role(self, emp_id: int, template: str):
        await self._api(
            "PUT", "/employee/entitlement/:grantEntitlementsByTemplate",
            params={"employeeId": emp_id, "template": template},
        )

    def _err(self, result: dict, step: str = "", **extra) -> dict:
        return {
            "success": False, "error": result.get("data", {}),
            "status_code": result.get("status_code"), "step": step, **extra,
        }

    async def _get_account_id(self, acc_num: int) -> int | None:
        if acc_num in self._account_cache:
            return self._account_cache[acc_num]
        accounts = self.ref.get("accounts", [])
        for acc in accounts:
            if acc.get("number") == acc_num:
                self._account_cache[acc_num] = acc["id"]
                return acc["id"]
        ar = await self._api("GET", "/ledger/account",
                             params={"number": acc_num, "fields": "id,number,name"})
        vals = self._vals(ar)
        if vals:
            self._account_cache[acc_num] = vals[0]["id"]
            return vals[0]["id"]
        return None

    # ─── create_employee ────────────────────────────────────────────────

    async def _do_create_employee(self, inp: dict) -> dict:
        role = inp.get("role")
        user_type = inp.get("userType", "STANDARD")

        if role:
            role = role.upper()
            if role in ELEVATED_ROLES:
                user_type = "EXTENDED"

        existing = await self._search_employee(inp["firstName"], inp["lastName"])
        if existing:
            eid = existing["id"]
            if role:
                if existing.get("userType") != "EXTENDED" and user_type == "EXTENDED":
                    await self._api("PUT", f"/employee/{eid}", body={
                        "id": eid, "version": existing.get("version", 1),
                        "userType": "EXTENDED",
                    })
                await self._grant_role(eid, role)
            return {"success": True, "employeeId": eid, "note": "existing employee found"}

        body: dict[str, Any] = {
            "firstName": inp["firstName"],
            "lastName": inp["lastName"],
            "userType": user_type,
        }
        for f in ("email", "phoneNumberMobile", "dateOfBirth",
                   "employeeNumber", "nationalIdentityNumber", "startDate"):
            if inp.get(f):
                body[f] = inp[f]
        if inp.get("allowInformationRegistration") is not None:
            body["allowInformationRegistration"] = inp["allowInformationRegistration"]
        if inp.get("address"):
            body["address"] = inp["address"]

        dept_inp = inp.get("department")
        if dept_inp:
            if dept_inp.get("id"):
                body["department"] = {"id": dept_inp["id"]}
            elif dept_inp.get("name"):
                dept_id = self._match_ref("departments", "name", dept_inp["name"])
                if dept_id:
                    body["department"] = {"id": dept_id}

        result = await self._api("POST", "/employee", body=body)

        if result.get("status_code") == 422:
            err = json.dumps(result.get("data", {})).lower()
            if "department" in err and "department" not in body:
                dept = self._ref_id("departments")
                if dept:
                    body["department"] = {"id": dept}
                    result = await self._api("POST", "/employee", body=body)

        if result.get("status_code") == 422 and body.get("email"):
            err = json.dumps(result.get("data", {})).lower()
            if "e-post" in err or "email" in err or "allerede" in err:
                sr = await self._api(
                    "GET", "/employee",
                    params={"email": body["email"],
                            "fields": "id,firstName,lastName,userType,version"},
                )
                vals = self._vals(sr)
                if vals:
                    eid = vals[0]["id"]
                    if role:
                        if vals[0].get("userType") != "EXTENDED" and user_type == "EXTENDED":
                            await self._api("PUT", f"/employee/{eid}", body={
                                "id": eid, "version": vals[0].get("version", 1),
                                "userType": "EXTENDED",
                            })
                        await self._grant_role(eid, role)
                    return {"success": True, "employeeId": eid, "note": "existing (email conflict)"}

        if result.get("status_code") == 422 and "department" not in body:
            dept = self._ref_id("departments")
            if dept:
                body["department"] = {"id": dept}
                result = await self._api("POST", "/employee", body=body)

        if self._ok(result):
            eid = self._id(result)
            if role and eid:
                await self._grant_role(eid, role)
            return {"success": True, "employeeId": eid}

        return self._err(result, "create")

    # ─── create_customer ────────────────────────────────────────────────

    async def _do_create_customer(self, inp: dict) -> dict:
        body: dict[str, Any] = {"name": inp["name"], "isCustomer": True}
        for f in ("email", "phoneNumber", "organizationNumber", "customerNumber",
                   "isPrivateIndividual", "invoiceSendMethod", "website", "description",
                   "invoiceEmail"):
            if inp.get(f) is not None:
                body[f] = inp[f]

        if inp.get("address"):
            a = inp["address"]
            body["postalAddress"] = {
                "addressLine1": a.get("addressLine1", ""),
                "postalCode": a.get("postalCode", ""),
                "city": a.get("city", ""),
                "country": {"id": 161},
            }

        result = await self._api("POST", "/customer", body=body)
        if not self._ok(result):
            return self._err(result, "customer")

        cid = self._id(result)
        contact_ids = []
        for c in inp.get("contacts", []):
            cb: dict[str, Any] = {"customer": {"id": cid}}
            for f in ("firstName", "lastName", "email", "phoneNumberMobile"):
                if c.get(f):
                    cb[f] = c[f]
            cr = await self._api("POST", "/contact", body=cb)
            if self._ok(cr):
                contact_ids.append(self._id(cr))

        return {"success": True, "customerId": cid, "contactIds": contact_ids}

    # ─── create_invoice_workflow ────────────────────────────────────────

    async def _do_create_invoice_workflow(self, inp: dict) -> dict:
        today = self.today
        steps: list[str] = []

        customer_id = inp.get("customerId")
        if not customer_id:
            cname = inp.get("customerName")
            if not cname:
                return {"success": False, "error": "customerName or customerId required"}

            existing = self._find_customer_in_ref(cname)
            if existing:
                customer_id = existing["id"]
                steps.append("customer (existing)")
            else:
                sr = await self._api("GET", "/customer", params={
                    "name": cname, "fields": "id,name"
                })
                vals = self._vals(sr)
                if vals:
                    customer_id = vals[0]["id"]
                    steps.append("customer (found)")

            if not customer_id:
                cb: dict[str, Any] = {"name": cname, "isCustomer": True}
                for src, dst in [("customerEmail", "email"), ("customerPhone", "phoneNumber"),
                                 ("customerOrgNumber", "organizationNumber")]:
                    if inp.get(src):
                        cb[dst] = inp[src]
                if inp.get("isPrivateIndividual"):
                    cb["isPrivateIndividual"] = True
                if inp.get("invoiceSendMethod"):
                    cb["invoiceSendMethod"] = inp["invoiceSendMethod"]
                if inp.get("customerAddress"):
                    a = inp["customerAddress"]
                    cb["postalAddress"] = {
                        "addressLine1": a.get("addressLine1", ""),
                        "postalCode": a.get("postalCode", ""),
                        "city": a.get("city", ""),
                        "country": {"id": 161},
                    }
                cr = await self._api("POST", "/customer", body=cb)
                if not self._ok(cr):
                    return self._err(cr, "customer", steps_completed=steps)
                customer_id = self._id(cr)
                steps.append("customer")

        for c in inp.get("contacts", []):
            cb2: dict[str, Any] = {"customer": {"id": customer_id}}
            for f in ("firstName", "lastName", "email", "phoneNumberMobile"):
                if c.get(f):
                    cb2[f] = c[f]
            await self._api("POST", "/contact", body=cb2)

        order_lines = []
        for line in inp.get("orderLines", []):
            ol: dict[str, Any] = {
                "description": line.get("description", "Item"),
                "count": line.get("count", 1),
            }
            vat_pct = line.get("vatPercent", 25)
            if line.get("unitPriceIncludingVat") is not None:
                excl = line["unitPriceIncludingVat"] / (1 + vat_pct / 100)
                ol["unitPriceExcludingVatCurrency"] = round(excl, 2)
            elif line.get("unitPriceExcludingVat") is not None:
                ol["unitPriceExcludingVatCurrency"] = line["unitPriceExcludingVat"]
            else:
                ol["unitPriceExcludingVatCurrency"] = 0
            ol["vatType"] = {"number": VAT_NUMBER_MAP.get(int(vat_pct), "3")}

            if line.get("productId"):
                ol["product"] = {"id": line["productId"]}
            elif line.get("productName"):
                pr = await self._api("POST", "/product", body={
                    "name": line["productName"],
                    "number": line.get("productNumber", ""),
                    "priceExcludingVatCurrency": ol.get("unitPriceExcludingVatCurrency", 0),
                })
                if self._ok(pr):
                    pid = self._id(pr)
                    if pid:
                        ol["product"] = {"id": pid}

            order_lines.append(ol)

        order_date = inp.get("orderDate", today)
        delivery_date = inp.get("deliveryDate", order_date)

        order_body: dict[str, Any] = {
            "customer": {"id": customer_id},
            "orderDate": order_date,
            "deliveryDate": delivery_date,
            "isPrioritizeAmountsIncludingVat": False,
            "orderLines": order_lines,
        }
        if inp.get("comment"):
            order_body["comment"] = inp["comment"]

        orr = await self._api("POST", "/order", body=order_body)
        if not self._ok(orr):
            return self._err(orr, "order", steps_completed=steps, customerId=customer_id)
        order_id = self._id(orr)
        steps.append("order")

        inv_date = inp.get("invoiceDate", today)
        due = inp.get("dueDate")
        if due:
            ir = await self._api("POST", "/invoice", body={
                "invoiceDate": inv_date, "invoiceDueDate": due,
                "orders": [{"id": order_id}],
            })
        else:
            ir = await self._api(
                "PUT", f"/order/{order_id}/:invoice",
                params={"invoiceDate": inv_date, "sendToCustomer": "false"},
            )
        if not self._ok(ir):
            return self._err(ir, "invoice", steps_completed=steps,
                             customerId=customer_id, orderId=order_id)
        invoice_id = self._id(ir)
        steps.append("invoice")

        if inp.get("registerPayment"):
            det = await self._api("GET", f"/invoice/{invoice_id}",
                                  params={"fields": "id,amount,amountOutstanding"})
            amount = 0
            if self._ok(det):
                v = det.get("data", {}).get("value", {})
                amount = v.get("amountOutstanding") or v.get("amount", 0)
            ptid = self._match_ref("invoice_payment_types", "description", "bank")
            pdate = inp.get("paymentDate", today)
            if ptid and amount:
                pr = await self._api(
                    "PUT", f"/invoice/{invoice_id}/:payment",
                    params={"paymentDate": pdate, "paymentTypeId": ptid, "paidAmount": amount},
                )
                if self._ok(pr):
                    steps.append("payment")
                else:
                    return {
                        "success": True, "steps_completed": steps,
                        "customerId": customer_id, "orderId": order_id,
                        "invoiceId": invoice_id, "paymentError": pr.get("data"),
                    }

        return {
            "success": True, "steps_completed": steps,
            "customerId": customer_id, "orderId": order_id, "invoiceId": invoice_id,
        }

    # ─── manage_travel_expense ──────────────────────────────────────────

    async def _do_manage_travel_expense(self, inp: dict) -> dict:
        action = inp.get("action", "create")

        if action == "delete":
            tid = inp.get("travelExpenseId")
            if not tid:
                eid = inp.get("employeeId")
                if eid:
                    sr = await self._api("GET", "/travelExpense",
                                         params={"employeeId": eid, "fields": "id,title"})
                    vs = self._vals(sr)
                    if vs:
                        tid = vs[0]["id"]
            if tid:
                dr = await self._api("DELETE", f"/travelExpense/{tid}")
                return {"success": self._ok(dr), "deletedId": tid}
            return {"success": False, "error": "Travel expense not found"}

        eid = inp.get("employeeId")
        if not eid and inp.get("employeeName"):
            parts = inp["employeeName"].split()
            first, last = parts[0], " ".join(parts[1:]) if len(parts) > 1 else ""
            emp = await self._search_employee(first, last)
            if emp:
                eid = emp["id"]
        if not eid:
            return {"success": False, "error": "employeeId required (or employeeName to search)"}

        steps: list[str] = []

        body: dict[str, Any] = {"employee": {"id": eid}}
        if inp.get("title"):
            body["title"] = inp["title"]

        td = inp.get("travelDetails")
        if td:
            dep_date = td.get("departureDate", self.today)
            ret_date = td.get("returnDate", dep_date)
            is_day_trip = td.get("isDayTrip")
            if is_day_trip is None:
                try:
                    d1 = datetime.strptime(dep_date, "%Y-%m-%d").date()
                    d2 = datetime.strptime(ret_date, "%Y-%m-%d").date()
                    is_day_trip = d1 == d2
                except (ValueError, TypeError):
                    is_day_trip = dep_date == ret_date

            body["travelDetails"] = {
                "isForeignTravel": td.get("isForeignTravel", False),
                "isDayTrip": is_day_trip,
                "departureDate": dep_date,
                "returnDate": ret_date,
                "departureFrom": td.get("departureFrom", ""),
                "destination": td.get("destination", ""),
            }

        result = await self._api("POST", "/travelExpense", body=body)
        if not self._ok(result):
            if "department" in json.dumps(result.get("data", {})).lower():
                dept = self._ref_id("departments")
                if dept:
                    body["department"] = {"id": dept}
                    result = await self._api("POST", "/travelExpense", body=body)
            if not self._ok(result):
                return self._err(result, "create_shell")
        tid = self._id(result)
        steps.append("shell")

        cost_ids: list[int | None] = []
        for cost in inp.get("costs", []):
            cat_id = self._match_ref("travel_cost_categories", "description",
                                     cost.get("category", ""))
            pay_id = self._match_ref("travel_payment_types", "description", "privat")
            cb: dict[str, Any] = {
                "travelExpense": {"id": tid},
                "date": cost.get("date", self.today),
                "currency": {"id": 1},
                "amountCurrencyIncVat": cost.get("amount", 0),
            }
            if cost.get("description"):
                cb["comment"] = cost["description"]
            if cat_id:
                cb["costCategory"] = {"id": cat_id}
            if pay_id:
                cb["paymentType"] = {"id": pay_id}
            cr = await self._api("POST", "/travelExpense/cost", body=cb)
            if self._ok(cr):
                cost_ids.append(self._id(cr))
                steps.append("cost")
            else:
                logger.warning("Cost creation failed: %s", cr)

        mileage_id = None
        m = inp.get("mileage")
        if m and td:
            mb: dict[str, Any] = {
                "travelExpense": {"id": tid},
                "date": m.get("date", td.get("departureDate", self.today)),
                "km": m.get("km", 0),
                "departureLocation": m.get("departureLocation", td.get("departureFrom", "")),
                "destination": m.get("destination", td.get("destination", "")),
            }
            if m.get("rateTypeId"):
                mb["rateType"] = {"id": m["rateTypeId"]}
            mr = await self._api("POST", "/travelExpense/mileageAllowance", body=mb)
            if self._ok(mr):
                mileage_id = self._id(mr)
                steps.append("mileage")
            else:
                logger.warning("Mileage creation failed: %s", mr)

        per_diem_id = None
        pd = inp.get("perDiem")
        if pd and td:
            rid = pd.get("rateCategoryId")
            if not rid:
                rid = self._match_ref("travel_rate_categories", "name",
                                      pd.get("rateCategoryName", ""))
            dep_date_str = td.get("departureDate", self.today)
            pdb: dict[str, Any] = {
                "travelExpense": {"id": tid},
                "location": pd.get("location", td.get("destination", "")),
                "overnightAccommodation": pd.get("overnightAccommodation", "NONE"),
                "startDate": pd.get("startDate", dep_date_str),
                "endDate": pd.get("endDate", td.get("returnDate", dep_date_str)),
            }
            if rid:
                pdb["rateCategory"] = {"id": rid}
            pdr = await self._api("POST", "/travelExpense/perDiemCompensation", body=pdb)
            if self._ok(pdr):
                per_diem_id = self._id(pdr)
                steps.append("perDiem")
            else:
                logger.warning("Per diem creation failed: %s", pdr)

        return {
            "success": True, "travelExpenseId": tid, "employeeId": eid,
            "costIds": cost_ids, "mileageId": mileage_id, "perDiemId": per_diem_id,
            "steps_completed": steps,
        }

    # ─── create_project_workflow ────────────────────────────────────────

    async def _do_create_project_workflow(self, inp: dict) -> dict:
        steps: list[str] = []

        manager_id = inp.get("projectManagerEmployeeId")
        if not manager_id and inp.get("projectManagerName"):
            parts = inp["projectManagerName"].split()
            first, last = parts[0], " ".join(parts[1:]) if len(parts) > 1 else ""

            emp = await self._search_employee(first, last)
            if emp:
                manager_id = emp["id"]
                if emp.get("userType") != "EXTENDED":
                    await self._api("PUT", f"/employee/{manager_id}", body={
                        "id": manager_id,
                        "version": emp.get("version", 1),
                        "userType": "EXTENDED",
                    })
            else:
                er = await self._do_create_employee({
                    "firstName": first, "lastName": last, "userType": "EXTENDED",
                })
                if er.get("success"):
                    manager_id = er["employeeId"]

        if manager_id:
            await self._grant_role(manager_id, "DEPARTMENT_LEADER")
            steps.append("manager")

        cats = self.ref.get("project_categories", [])
        cat_id = None

        cat_name = inp.get("categoryName")
        if cat_name:
            for cat in cats:
                if cat_name.lower() in str(cat.get("name", "")).lower():
                    cat_id = cat["id"]
                    break

        if not cat_id and cats:
            cat_id = cats[0]["id"]

        if not cat_id:
            cname = cat_name or "General"
            catr = await self._api("POST", "/project/category",
                                   body={"name": cname, "number": "1"})
            if self._ok(catr):
                cat_id = self._id(catr)
        steps.append("category")

        customer_id = inp.get("customerId")
        if not customer_id and inp.get("customerName"):
            existing = self._find_customer_in_ref(inp["customerName"])
            if existing:
                customer_id = existing["id"]
            else:
                custr = await self._api("POST", "/customer",
                                        body={"name": inp["customerName"], "isCustomer": True})
                if self._ok(custr):
                    customer_id = self._id(custr)
            steps.append("customer")

        is_internal = inp.get("isInternal")
        if is_internal is None:
            is_internal = customer_id is None

        body: dict[str, Any] = {"name": inp["name"], "isInternal": is_internal}
        if cat_id:
            body["projectCategory"] = {"id": cat_id}
        if manager_id:
            body["projectManager"] = {"id": manager_id}
        if customer_id:
            body["customer"] = {"id": customer_id}
        for f in ("startDate", "endDate", "description", "number"):
            if inp.get(f):
                body[f] = inp[f]
        if inp.get("fixedPrice") is not None:
            body["fixedprice"] = inp["fixedPrice"]

        dept = self._ref_id("departments")
        if dept:
            body["department"] = {"id": dept}

        result = await self._api("POST", "/project", body=body)
        if not self._ok(result):
            return self._err(result, "project", steps_completed=steps, managerId=manager_id)

        return {
            "success": True, "projectId": self._id(result),
            "managerId": manager_id, "categoryId": cat_id,
            "steps_completed": steps,
        }

    # ─── create_voucher ─────────────────────────────────────────────────

    async def _do_create_voucher(self, inp: dict) -> dict:
        d = inp.get("date", self.today)

        total = sum(p.get("amount", 0) for p in inp.get("postings", []))
        if abs(total) > 0.01:
            return {"success": False, "error": f"Postings don't balance: sum={total}. Positive=debit, negative=credit, must sum to 0."}

        postings = []
        for p in inp.get("postings", []):
            acc_num = p["accountNumber"]
            acc_id = await self._get_account_id(acc_num)
            if not acc_id:
                return {"success": False, "error": f"Account {acc_num} not found"}

            posting: dict[str, Any] = {
                "date": d,
                "account": {"id": acc_id},
                "amountGross": p["amount"],
                "amountGrossCurrency": p["amount"],
                "currency": {"id": 1},
            }
            if p.get("description"):
                posting["description"] = p["description"]
            if p.get("customerId"):
                posting["customer"] = {"id": p["customerId"]}
            if p.get("supplierId"):
                posting["supplier"] = {"id": p["supplierId"]}
            if p.get("employeeId"):
                posting["employee"] = {"id": p["employeeId"]}
            if p.get("vatTypeNumber"):
                posting["vatType"] = {"number": p["vatTypeNumber"]}
            postings.append(posting)

        endpoint = "/ledger/voucher"
        if inp.get("useOpeningBalance"):
            endpoint = "/ledger/voucher/openingBalance"

        result = await self._api("POST", endpoint, body={
            "date": d, "description": inp.get("description", ""), "postings": postings,
        })
        if self._ok(result):
            return {"success": True, "voucherId": self._id(result)}

        err_text = json.dumps(result.get("data", {})).lower()
        if "opening" in err_text or "åpning" in err_text:
            if endpoint != "/ledger/voucher/openingBalance":
                result = await self._api("POST", "/ledger/voucher/openingBalance", body={
                    "date": d, "description": inp.get("description", ""), "postings": postings,
                })
                if self._ok(result):
                    return {"success": True, "voucherId": self._id(result)}

        return self._err(result, "voucher")

    # ─── tripletex_api (raw fallback) ───────────────────────────────────

    async def _do_tripletex_api(self, inp: dict) -> dict:
        return await self._api(
            inp.get("method", "GET"),
            inp.get("path", "/"),
            params=inp.get("params"),
            body=inp.get("json_body"),
        )
