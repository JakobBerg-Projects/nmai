"""High-level Tripletex tools with validation and multi-step workflow handling."""

import json
import logging
from datetime import date
from typing import Any

logger = logging.getLogger("tools")

VAT_NUMBER_MAP = {25: "3", 12: "33", 15: "31", 0: "5"}
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
                "address": {
                    "type": "object",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
                    },
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
                "customerAddress": {
                    "type": "object",
                    "properties": {
                        "addressLine1": {"type": "string"},
                        "postalCode": {"type": "string"},
                        "city": {"type": "string"},
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
            "Use for: reiseregning, ansattutlegg, travel expense, Reisekosten."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "delete"], "description": "Default: create"},
                "employeeId": {"type": "integer"},
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
                        "isDayTrip": {"type": "boolean"},
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
                    },
                },
                "perDiem": {
                    "type": "object",
                    "properties": {
                        "rateCategoryName": {
                            "type": "string",
                            "description": "e.g. 'Dagsreise over 12 timer'",
                        },
                        "location": {"type": "string"},
                        "overnightAccommodation": {
                            "type": "string",
                            "enum": ["NONE", "HOTEL", "OTHER"],
                        },
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
            "Use for: bilag, voucher, bokføring, opening balance, åpningsbalanse."
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
                            "customerId": {"type": "integer"},
                            "supplierId": {"type": "integer"},
                            "employeeId": {"type": "integer"},
                        },
                        "required": ["accountNumber", "amount"],
                    },
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
            "and any endpoint the high-level tools don't support."
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
        sl = search.lower()
        for item in items:
            if sl in str(item.get(field, "")).lower():
                return item["id"]
        return items[0]["id"] if items else None

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

    # ─── create_employee ────────────────────────────────────────────────

    async def _do_create_employee(self, inp: dict) -> dict:
        role = inp.get("role")
        user_type = inp.get("userType", "STANDARD")

        if role:
            role = role.upper()
            if role in ELEVATED_ROLES:
                user_type = "EXTENDED"

        body: dict[str, Any] = {
            "firstName": inp["firstName"],
            "lastName": inp["lastName"],
            "userType": user_type,
        }
        for f in ("email", "phoneNumberMobile", "dateOfBirth",
                   "employeeNumber", "nationalIdentityNumber"):
            if inp.get(f):
                body[f] = inp[f]
        if inp.get("address"):
            body["address"] = inp["address"]

        result = await self._api("POST", "/employee", body=body)

        if result.get("status_code") == 422:
            err = json.dumps(result.get("data", {})).lower()
            if "department" in err:
                dept = self._ref_id("departments")
                if dept:
                    body["department"] = {"id": dept}
                    result = await self._api("POST", "/employee", body=body)

        if result.get("status_code") == 422 and body.get("email"):
            err = json.dumps(result.get("data", {})).lower()
            if "e-post" in err or "email" in err:
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
                    return {"success": True, "employeeId": eid, "note": "existing"}

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
                   "isPrivateIndividual", "invoiceSendMethod", "website", "description"):
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
            cb: dict[str, Any] = {"name": cname, "isCustomer": True}
            for src, dst in [("customerEmail", "email"), ("customerPhone", "phoneNumber"),
                             ("customerOrgNumber", "organizationNumber")]:
                if inp.get(src):
                    cb[dst] = inp[src]
            if inp.get("isPrivateIndividual"):
                cb["isPrivateIndividual"] = True
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
            order_lines.append(ol)

        order_date = inp.get("orderDate", today)
        delivery_date = inp.get("deliveryDate", order_date)
        orr = await self._api("POST", "/order", body={
            "customer": {"id": customer_id},
            "orderDate": order_date,
            "deliveryDate": delivery_date,
            "isPrioritizeAmountsIncludingVat": False,
            "orderLines": order_lines,
        })
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

        eid = inp["employeeId"]
        steps: list[str] = []

        body: dict[str, Any] = {"employee": {"id": eid}}
        if inp.get("title"):
            body["title"] = inp["title"]

        td = inp.get("travelDetails")
        if td:
            body["travelDetails"] = {
                "isForeignTravel": td.get("isForeignTravel", False),
                "isDayTrip": td.get("isDayTrip", True),
                "departureDate": td.get("departureDate", self.today),
                "returnDate": td.get("returnDate", td.get("departureDate", self.today)),
                "departureFrom": td.get("departureFrom", ""),
                "destination": td.get("destination", ""),
            }

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
            if cat_id:
                cb["costCategory"] = {"id": cat_id}
            if pay_id:
                cb["paymentType"] = {"id": pay_id}
            cr = await self._api("POST", "/travelExpense/cost", body=cb)
            if self._ok(cr):
                cost_ids.append(self._id(cr))
                steps.append("cost")

        mileage_id = None
        m = inp.get("mileage")
        if m:
            mr = await self._api("POST", "/travelExpense/mileageAllowance", body={
                "travelExpense": {"id": tid},
                "date": m.get("date", self.today),
                "km": m.get("km", 0),
                "departureLocation": m.get("departureLocation", ""),
                "destination": m.get("destination", ""),
            })
            if self._ok(mr):
                mileage_id = self._id(mr)
                steps.append("mileage")

        per_diem_id = None
        pd = inp.get("perDiem")
        if pd:
            rid = self._match_ref("travel_rate_categories", "name",
                                  pd.get("rateCategoryName", ""))
            pdb: dict[str, Any] = {
                "travelExpense": {"id": tid},
                "location": pd.get("location", ""),
                "overnightAccommodation": pd.get("overnightAccommodation", "NONE"),
            }
            if rid:
                pdb["rateCategory"] = {"id": rid}
            pdr = await self._api("POST", "/travelExpense/perDiemCompensation", body=pdb)
            if self._ok(pdr):
                per_diem_id = self._id(pdr)
                steps.append("perDiem")

        return {
            "success": True, "travelExpenseId": tid,
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

            sr = await self._api("GET", "/employee", params={
                "firstName": first, "lastName": last,
                "fields": "id,firstName,lastName,userType,version",
            })
            vals = self._vals(sr)
            if vals:
                manager_id = vals[0]["id"]
                if vals[0].get("userType") != "EXTENDED":
                    await self._api("PUT", f"/employee/{manager_id}", body={
                        "id": manager_id,
                        "version": vals[0].get("version", 1),
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
        cat_id = cats[0]["id"] if cats else None
        if not cat_id:
            cname = inp.get("categoryName", "General")
            catr = await self._api("POST", "/project/category",
                                   body={"name": cname, "number": "1"})
            if self._ok(catr):
                cat_id = self._id(catr)
        steps.append("category")

        customer_id = inp.get("customerId")
        if not customer_id and inp.get("customerName"):
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
            return {"success": False, "error": f"Postings don't balance: sum={total}"}

        postings = []
        for p in inp.get("postings", []):
            acc_num = p["accountNumber"]
            ar = await self._api("GET", "/ledger/account",
                                 params={"number": acc_num, "fields": "id,number,name"})
            vals = self._vals(ar)
            if not vals:
                return {"success": False, "error": f"Account {acc_num} not found"}

            posting: dict[str, Any] = {
                "date": d,
                "account": {"id": vals[0]["id"]},
                "amountGross": p["amount"],
                "amountGrossCurrency": p["amount"],
                "currency": {"id": 1},
            }
            if p.get("customerId"):
                posting["customer"] = {"id": p["customerId"]}
            if p.get("supplierId"):
                posting["supplier"] = {"id": p["supplierId"]}
            if p.get("employeeId"):
                posting["employee"] = {"id": p["employeeId"]}
            postings.append(posting)

        result = await self._api("POST", "/ledger/voucher", body={
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
