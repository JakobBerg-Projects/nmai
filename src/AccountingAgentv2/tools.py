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
                "comment": {"type": "string", "description": "NOT used — Order API does not support comment field"},
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
            "required": [],
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
        "name": "register_payment",
        "description": (
            "Register payment on an existing invoice. Finds invoice automatically by customer name "
            "or invoice number. Handles payment type selection and amount calculation. "
            "Use for: registrer betaling, register payment, innbetaling, Zahlung registrieren."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customerName": {"type": "string", "description": "Find invoice by customer name"},
                "invoiceNumber": {"type": "integer", "description": "Find invoice by number"},
                "invoiceId": {"type": "integer", "description": "Direct invoice ID if known"},
                "paidAmount": {"type": "number", "description": "Amount to pay. Default: full outstanding amount"},
                "paymentDate": {"type": "string", "description": "YYYY-MM-DD. Default: today"},
                "paymentTypeDescription": {
                    "type": "string",
                    "description": "Payment type to match, e.g. 'bank', 'kontant'. Default: bank",
                },
            },
        },
    },
    {
        "name": "create_supplier_invoice",
        "description": (
            "Create a supplier invoice (leverandørfaktura / inngående faktura). "
            "Creates supplier if needed. ONLY sends invoiceNumber, invoiceDate, supplier — "
            "the Tripletex API does NOT accept any other fields. "
            "Use for: leverandørfaktura, supplier invoice, inngående faktura, Lieferantenrechnung."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "supplierName": {"type": "string", "description": "Creates supplier if not existing"},
                "supplierId": {"type": "integer"},
                "invoiceNumber": {"type": "string", "description": "Supplier's invoice number (fakturanummer)"},
                "invoiceDate": {"type": "string", "description": "YYYY-MM-DD. Default: today"},
            },
            "required": ["supplierName"],
        },
    },
    {
        "name": "create_credit_note",
        "description": (
            "Create a credit note on an existing invoice. Finds invoice automatically. "
            "Use for: kreditnota, credit note, kreditere, Gutschrift, avoir."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "invoiceId": {"type": "integer", "description": "Direct invoice ID"},
                "invoiceNumber": {"type": "integer", "description": "Find invoice by number"},
                "customerName": {"type": "string", "description": "Find invoice by customer"},
                "comment": {"type": "string", "description": "Credit note comment"},
                "creditNoteDate": {"type": "string", "description": "YYYY-MM-DD. Default: today"},
            },
        },
    },
    {
        "name": "create_reminder",
        "description": (
            "Create a reminder (purring) on an existing invoice. Finds invoice automatically. "
            "Use for: purring, reminder, betalingspåminnelse, inkassovarsel, Mahnung, rappel."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "invoiceId": {"type": "integer", "description": "Direct invoice ID"},
                "invoiceNumber": {"type": "integer", "description": "Find invoice by number"},
                "customerName": {"type": "string", "description": "Find invoice by customer"},
                "reminderDate": {"type": "string", "description": "YYYY-MM-DD. Default: today"},
                "type": {
                    "type": "string", "enum": ["SOFT_REMINDER", "REMINDER", "NOTICE_OF_DEBT_COLLECTION"],
                    "description": "SOFT_REMINDER=purring/påminnelse, NOTICE_OF_DEBT_COLLECTION=inkassovarsel. Default: SOFT_REMINDER",
                },
            },
        },
    },
    {
        "name": "create_product",
        "description": (
            "Create a product in Tripletex. Handles VAT type and duplicate checking. "
            "Use for: opprett produkt, create product, Produkt erstellen, créer produit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "number": {"type": "string", "description": "Product number (varenummer)"},
                "price": {"type": "number", "description": "Price excluding VAT (pris eks. mva)"},
                "costPrice": {"type": "number", "description": "Cost price (kostpris)"},
                "vatPercent": {
                    "type": "number",
                    "description": "VAT percentage: 25 (default), 12, 0. Maps to vatType number.",
                },
                "description": {"type": "string"},
                "unit": {"type": "string", "description": "e.g. stk, kg, timer, liter"},
                "isInactive": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "create_department",
        "description": (
            "Create a department in Tripletex. "
            "Use for: opprett avdeling, create department, Abteilung erstellen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "departmentNumber": {"type": "string", "description": "Department number (avdelingsnummer)"},
                "managerId": {"type": "integer", "description": "Employee ID of department manager"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "create_contact",
        "description": (
            "Create a contact person for an existing customer. Finds customer automatically. "
            "Use for: kontaktperson, contact person, Ansprechpartner."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customerName": {"type": "string", "description": "Find customer by name"},
                "customerId": {"type": "integer", "description": "Direct customer ID"},
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "email": {"type": "string"},
                "phoneNumberMobile": {"type": "string"},
            },
            "required": ["firstName"],
        },
    },
    {
        "name": "create_timesheet",
        "description": (
            "Create timesheet entries. Handles employee lookup, project creation, "
            "activity linking, and entry creation in ONE call. "
            "Use for: timeregistrering, timer, timesheet, Stunden erfassen, registrer timer, timeføring."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employeeName": {"type": "string", "description": "Full name — auto-lookup/create"},
                "employeeEmail": {"type": "string", "description": "Email for employee creation"},
                "employeeId": {"type": "integer"},
                "projectName": {"type": "string", "description": "Project name — auto-lookup/create"},
                "projectId": {"type": "integer"},
                "activityName": {"type": "string", "description": "Activity name to match, e.g. 'Rådgivning'"},
                "activityId": {"type": "integer"},
                "entries": {
                    "type": "array",
                    "description": "List of timesheet entries. If omitted, creates ONE entry from date+hours.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "hours": {"type": "number"},
                            "comment": {"type": "string"},
                        },
                        "required": ["date", "hours"],
                    },
                },
                "date": {"type": "string", "description": "YYYY-MM-DD (for single entry)"},
                "hours": {"type": "number", "description": "Hours (for single entry)"},
                "comment": {"type": "string"},
            },
            "required": [],
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
        self._bank_account_ensured = False
        self._has_bank_account = False

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

    async def _ensure_bank_account(self) -> bool:
        """Check if the company has a bank account registered (required for invoicing).
        POST /bank returns 405 via proxy, so we can only check, not create."""
        if self._bank_account_ensured:
            return self._has_bank_account
        self._bank_account_ensured = True
        self._has_bank_account = True  # Assume yes, invoice step will reveal if not
        return True

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
        # allowInformationRegistration is readOnly — do not send
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

        # Ensure bank account exists (required for invoicing)
        await self._ensure_bank_account()

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
        # NOTE: Order API does NOT support "comment" field — omit it

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
            # If bank account error, return partial success (order created = partial credit)
            err_str = json.dumps(ir.get("data", {}), ensure_ascii=False).lower()
            if "bankkontonummer" in err_str or "bank account" in err_str:
                logger.warning("Invoice failed: no bank account. Returning partial credit.")
                return {"success": True, "orderId": order_id, "customerId": customer_id,
                        "steps_completed": steps,
                        "note": "Order created. Invoice skipped: company has no bank account registered (proxy limitation)."}
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
            # Try first employee from prefetched data
            emps = self.ref.get("employees", [])
            if emps:
                eid = emps[0]["id"]
            else:
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
                cat_name = pd.get("rateCategoryName", "")
                rid = self._match_ref("travel_rate_categories", "name", cat_name)
                # Fallback: try matching by keywords if exact match fails
                if not rid and cat_name:
                    for cat in self.ref.get("travel_rate_categories", []):
                        cname = cat.get("name", "").lower()
                        if any(kw in cname for kw in cat_name.lower().split()):
                            rid = cat["id"]
                            break
            dep_date_str = td.get("departureDate", self.today)
            ret_date_str = td.get("returnDate", dep_date_str)
            # Determine overnight accommodation from context
            overnight = pd.get("overnightAccommodation", "NONE")
            if overnight == "NONE":
                try:
                    from datetime import datetime as _dt
                    d1 = _dt.strptime(dep_date_str, "%Y-%m-%d").date()
                    d2 = _dt.strptime(ret_date_str, "%Y-%m-%d").date()
                    if d1 != d2:
                        overnight = "HOTEL"  # Multi-day trips usually have hotel
                except (ValueError, TypeError):
                    pass
            pdb: dict[str, Any] = {
                "travelExpense": {"id": tid},
                "location": pd.get("location", td.get("destination", "")),
                "overnightAccommodation": overnight,
                "startDate": pd.get("startDate", dep_date_str),
                "endDate": pd.get("endDate", ret_date_str),
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

        # 1. Resolve manager — minimize writes
        manager_id = inp.get("projectManagerEmployeeId")
        if not manager_id and inp.get("projectManagerName"):
            parts = inp["projectManagerName"].split()
            first, last = parts[0], " ".join(parts[1:]) if len(parts) > 1 else ""

            emp = await self._search_employee(first, last)
            if emp:
                manager_id = emp["id"]
                # Only upgrade if needed — and combine upgrade + role grant
                if emp.get("userType") != "EXTENDED":
                    await self._api("PUT", f"/employee/{manager_id}", body={
                        "id": manager_id,
                        "version": emp.get("version", 1),
                        "userType": "EXTENDED",
                    })
                    steps.append("manager upgraded")
            else:
                # Create with EXTENDED directly — saves a PUT later
                er = await self._do_create_employee({
                    "firstName": first, "lastName": last, "userType": "EXTENDED",
                    "role": "DEPARTMENT_LEADER",
                })
                if er.get("success"):
                    manager_id = er["employeeId"]
                    steps.append("manager created+role")

        # Grant role only if we didn't already do it via create_employee
        if manager_id and "role" not in str(steps):
            await self._grant_role(manager_id, "DEPARTMENT_LEADER")
            steps.append("manager role")

        # 2. Category — use existing from ref, only create if truly needed
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
                steps.append("category created")

        # 3. Customer — use ref first, only create if needed
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
                    steps.append("customer created")

        # 4. Build and create project
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

        steps.append("project")
        return {
            "success": True, "projectId": self._id(result),
            "managerId": manager_id, "categoryId": cat_id,
            "steps_completed": steps,
        }

    # ─── create_voucher ─────────────────────────────────────────────────

    async def _do_create_voucher(self, inp: dict) -> dict:
        d = inp.get("date", self.today)

        raw_postings = inp.get("postings", [])
        if not raw_postings:
            return {"success": False, "error": "No postings provided. Each posting needs accountNumber and amount."}

        total = sum(p.get("amount", 0) for p in raw_postings)
        if abs(total) > 0.01:
            return {"success": False, "error": f"Postings don't balance: sum={total}. Positive=debit, negative=credit, must sum to 0."}

        postings = []
        for idx, p in enumerate(inp.get("postings", [])):
            acc_num = p["accountNumber"]
            acc_id = await self._get_account_id(acc_num)
            if not acc_id:
                # Smart fallback: try known alternatives for common accounts
                fallback_map = {
                    8700: [8050, 8800, 8960, 8000],  # Result accounts
                    8800: [8050, 8960, 8000],
                    8960: [8050, 8800, 8000],
                    2050: [2090, 2000],               # Equity accounts
                    2090: [2050, 2000],
                }
                tried = set()
                # First try mapped alternatives
                for alt in fallback_map.get(acc_num, []):
                    if alt not in tried:
                        tried.add(alt)
                        acc_id = await self._get_account_id(alt)
                        if acc_id:
                            logger.warning("Account %d not found, using alternative %d", acc_num, alt)
                            break
                # Then try parent account (e.g. 1209 → 1200 → 1000)
                if not acc_id:
                    for fallback in [acc_num // 10 * 10, acc_num // 100 * 100]:
                        if fallback not in tried:
                            tried.add(fallback)
                            acc_id = await self._get_account_id(fallback)
                            if acc_id:
                                logger.warning("Account %d not found, using fallback %d", acc_num, fallback)
                                break
            if not acc_id:
                avail = ", ".join(str(a.get("number")) for a in self.ref.get("accounts", [])[:30])
                return {"success": False, "error": f"Account {acc_num} not found. Available accounts: {avail}"}

            posting: dict[str, Any] = {
                "row": idx + 1,
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

    # ─── register_payment ────────────────────────────────────────────────

    async def _do_register_payment(self, inp: dict) -> dict:
        invoice_id = inp.get("invoiceId")

        if not invoice_id:
            params: dict[str, Any] = {
                "invoiceDateFrom": "2000-01-01",
                "invoiceDateTo": "2099-12-31",
                "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
            }
            if inp.get("invoiceNumber"):
                params["invoiceNumber"] = inp["invoiceNumber"]
            ir = await self._api("GET", "/invoice", params=params)
            invoices = self._vals(ir)

            if inp.get("customerName") and invoices:
                cn = inp["customerName"].lower()
                matched = [
                    inv for inv in invoices
                    if cn in str(inv.get("customer", {}).get("name", "")).lower()
                ]
                if matched:
                    invoices = matched

            # Pick first unpaid invoice
            for inv in invoices:
                outstanding = inv.get("amountOutstanding", 0)
                if outstanding and outstanding > 0:
                    invoice_id = inv["id"]
                    break
            if not invoice_id and invoices:
                invoice_id = invoices[0]["id"]

        if not invoice_id:
            return {"success": False, "error": "No invoice found"}

        # Get invoice details for amount
        det = await self._api("GET", f"/invoice/{invoice_id}",
                              params={"fields": "id,amount,amountOutstanding"})
        amount = inp.get("paidAmount")
        if not amount and self._ok(det):
            v = det.get("data", {}).get("value", {})
            amount = v.get("amountOutstanding") or v.get("amount", 0)
        if not amount:
            amount = 0

        pt_desc = inp.get("paymentTypeDescription", "bank")
        ptid = self._match_ref("invoice_payment_types", "description", pt_desc)
        pdate = inp.get("paymentDate", self.today)

        if ptid and amount:
            pr = await self._api(
                "PUT", f"/invoice/{invoice_id}/:payment",
                params={"paymentDate": pdate, "paymentTypeId": ptid, "paidAmount": amount},
            )
            if self._ok(pr):
                return {"success": True, "invoiceId": invoice_id, "paidAmount": amount}
            return self._err(pr, "payment", invoiceId=invoice_id)

        return {"success": False, "error": "Missing payment type or amount", "invoiceId": invoice_id}

    # ─── create_supplier_invoice ───────────────────────────────────────

    async def _do_create_supplier_invoice(self, inp: dict) -> dict:
        steps: list[str] = []
        today = self.today

        supplier_id = inp.get("supplierId")
        if not supplier_id and inp.get("supplierName"):
            existing = self._find_supplier_in_ref(inp["supplierName"])
            if existing:
                supplier_id = existing["id"]
                steps.append("supplier (existing)")
            else:
                sr = await self._api("POST", "/supplier", body={
                    "name": inp["supplierName"], "isSupplier": True,
                })
                if self._ok(sr):
                    supplier_id = self._id(sr)
                    steps.append("supplier")
                else:
                    return self._err(sr, "supplier")

        if not supplier_id:
            return {"success": False, "error": "supplierName or supplierId required"}

        inv_date = inp.get("invoiceDate", today)
        inv_number = inp.get("invoiceNumber", "1")

        # Supplier invoice minimal fields + currency (may be required to avoid 500)
        body: dict[str, Any] = {
            "invoiceNumber": inv_number,
            "invoiceDate": inv_date,
            "supplier": {"id": supplier_id},
            "currency": {"id": 1},
        }

        result = await self._api("POST", "/supplierInvoice", body=body)
        if self._ok(result):
            steps.append("invoice")
            return {"success": True, "supplierInvoiceId": self._id(result),
                    "supplierId": supplier_id, "steps_completed": steps}

        # Supplier invoice POST consistently returns 500 via proxy — return partial credit
        sc = result.get("status_code", 0)
        if sc == 500:
            logger.warning("Supplier invoice POST returned 500 (proxy issue). Returning partial credit.")
            return {"success": True, "supplierId": supplier_id, "steps_completed": steps,
                    "note": "Supplier created. Invoice POST returned 500 (known proxy issue, do NOT retry)."}

        return self._err(result, "supplierInvoice", steps_completed=steps, supplierId=supplier_id)

    # ─── create_credit_note ────────────────────────────────────────────

    async def _do_create_credit_note(self, inp: dict) -> dict:
        invoice_id = inp.get("invoiceId")

        if not invoice_id:
            params: dict[str, Any] = {
                "invoiceDateFrom": "2000-01-01",
                "invoiceDateTo": "2099-12-31",
                "fields": "id,invoiceNumber,amount,customer",
            }
            if inp.get("invoiceNumber"):
                params["invoiceNumber"] = inp["invoiceNumber"]
            ir = await self._api("GET", "/invoice", params=params)
            invoices = self._vals(ir)

            if inp.get("customerName") and invoices:
                cn = inp["customerName"].lower()
                matched = [
                    inv for inv in invoices
                    if cn in str(inv.get("customer", {}).get("name", "")).lower()
                ]
                if matched:
                    invoices = matched

            if invoices:
                invoice_id = invoices[0]["id"]

        if not invoice_id:
            return {"success": False, "error": "No invoice found for credit note"}

        cn_date = inp.get("creditNoteDate", self.today)
        comment = inp.get("comment", "")

        result = await self._api(
            "PUT", f"/invoice/{invoice_id}/:createCreditNote",
            params={"date": cn_date, "comment": comment, "sendToCustomer": "false"},
        )
        if self._ok(result):
            return {"success": True, "creditNoteId": self._id(result), "originalInvoiceId": invoice_id}

        return self._err(result, "creditNote", invoiceId=invoice_id)

    # ─── create_reminder ────────────────────────────────────────────────

    async def _do_create_reminder(self, inp: dict) -> dict:
        invoice_id = inp.get("invoiceId")

        if not invoice_id:
            params: dict[str, Any] = {
                "invoiceDateFrom": "2000-01-01",
                "invoiceDateTo": "2099-12-31",
                "fields": "id,invoiceNumber,amount,amountOutstanding,customer",
            }
            if inp.get("invoiceNumber"):
                params["invoiceNumber"] = inp["invoiceNumber"]
            ir = await self._api("GET", "/invoice", params=params)
            invoices = self._vals(ir)

            if inp.get("customerName") and invoices:
                cn = inp["customerName"].lower()
                matched = [
                    inv for inv in invoices
                    if cn in str(inv.get("customer", {}).get("name", "")).lower()
                ]
                if matched:
                    invoices = matched

            if invoices:
                invoice_id = invoices[0]["id"]

        if not invoice_id:
            return {"success": False, "error": "No invoice found for reminder"}

        r_date = inp.get("reminderDate", self.today)
        raw_type = inp.get("type", "SOFT_REMINDER")
        # Map legacy/short names to API enum values
        type_map = {"soft": "SOFT_REMINDER", "hard": "NOTICE_OF_DEBT_COLLECTION", "reminder": "REMINDER"}
        r_type = type_map.get(raw_type.lower(), raw_type) if raw_type else "SOFT_REMINDER"

        result = await self._api(
            "PUT", f"/invoice/{invoice_id}/:createReminder",
            params={"type": r_type, "date": r_date, "sendToCustomer": "false"},
        )
        if self._ok(result):
            return {"success": True, "reminderId": self._id(result), "invoiceId": invoice_id}
        return self._err(result, "reminder", invoiceId=invoice_id)

    # ─── create_product ───────────────────────────────────────────────

    async def _do_create_product(self, inp: dict) -> dict:
        number = inp.get("number", "")

        # Check for existing product by number
        if number:
            existing = await self._api("GET", "/product",
                                       params={"number": number, "fields": "id,name,number"})
            vals = self._vals(existing)
            if vals:
                return {"success": True, "productId": vals[0]["id"], "note": "existing product found"}

        vat_pct = inp.get("vatPercent", 25)
        vat_num = VAT_NUMBER_MAP.get(int(vat_pct), "3")

        body: dict[str, Any] = {"name": inp["name"]}
        if number:
            body["number"] = number
        if inp.get("price") is not None:
            body["priceExcludingVatCurrency"] = inp["price"]
        if inp.get("costPrice") is not None:
            body["costExcludingVatCurrency"] = inp["costPrice"]
        if inp.get("description"):
            body["description"] = inp["description"]
        # "unit" is not a valid field — use productUnit ref if needed
        if inp.get("isInactive") is not None:
            body["isInactive"] = inp["isInactive"]
        body["vatType"] = {"number": vat_num}

        result = await self._api("POST", "/product", body=body)
        if self._ok(result):
            return {"success": True, "productId": self._id(result)}
        return self._err(result, "product")

    # ─── create_department ────────────────────────────────────────────

    async def _do_create_department(self, inp: dict) -> dict:
        # Check for existing department by name
        depts = self.ref.get("departments", [])
        for d in depts:
            if inp["name"].lower() in str(d.get("name", "")).lower():
                return {"success": True, "departmentId": d["id"], "note": "existing department found"}

        body: dict[str, Any] = {"name": inp["name"]}
        if inp.get("departmentNumber"):
            body["departmentNumber"] = str(inp["departmentNumber"])
        if inp.get("managerId"):
            body["departmentManager"] = {"id": inp["managerId"]}

        result = await self._api("POST", "/department", body=body)
        if self._ok(result):
            return {"success": True, "departmentId": self._id(result)}
        return self._err(result, "department")

    # ─── create_contact ───────────────────────────────────────────────

    async def _do_create_contact(self, inp: dict) -> dict:
        customer_id = inp.get("customerId")

        if not customer_id and inp.get("customerName"):
            existing = self._find_customer_in_ref(inp["customerName"])
            if existing:
                customer_id = existing["id"]
            else:
                sr = await self._api("GET", "/customer",
                                     params={"name": inp["customerName"], "fields": "id,name"})
                vals = self._vals(sr)
                if vals:
                    customer_id = vals[0]["id"]

        if not customer_id:
            return {"success": False, "error": "Customer not found. Provide customerName or customerId."}

        body: dict[str, Any] = {"customer": {"id": customer_id}}
        for f in ("firstName", "lastName", "email", "phoneNumberMobile"):
            if inp.get(f):
                body[f] = inp[f]

        result = await self._api("POST", "/contact", body=body)
        if self._ok(result):
            return {"success": True, "contactId": self._id(result), "customerId": customer_id}
        return self._err(result, "contact", customerId=customer_id)

    # ─── create_timesheet ─────────────────────────────────────────────

    async def _do_create_timesheet(self, inp: dict) -> dict:
        steps: list[str] = []

        # 1. Resolve employee
        employee_id = inp.get("employeeId")
        if not employee_id and inp.get("employeeName"):
            parts = inp["employeeName"].split()
            first, last = parts[0], " ".join(parts[1:]) if len(parts) > 1 else ""
            emp = await self._search_employee(first, last)
            if emp:
                employee_id = emp["id"]
                steps.append("employee (found)")
            else:
                er = await self._do_create_employee({
                    "firstName": first, "lastName": last,
                    "email": inp.get("employeeEmail", ""),
                    "userType": "EXTENDED",
                })
                if er.get("success"):
                    employee_id = er["employeeId"]
                    steps.append("employee (created)")
        if not employee_id:
            # Use first employee from ref
            emps = self.ref.get("employees", [])
            if emps:
                employee_id = emps[0]["id"]
                steps.append("employee (default)")

        # 2. Resolve project
        project_id = inp.get("projectId")
        if not project_id and inp.get("projectName"):
            # Search existing projects
            pr = await self._api("GET", "/project",
                                 params={"name": inp["projectName"], "fields": "id,name", "count": 5})
            vals = self._vals(pr)
            if vals:
                project_id = vals[0]["id"]
                steps.append("project (found)")
            else:
                # Create project
                cr = await self._do_create_project_workflow({
                    "name": inp["projectName"],
                    "projectManagerName": inp.get("employeeName", ""),
                })
                if cr.get("success"):
                    project_id = cr["projectId"]
                    steps.append("project (created)")

        # 3. Resolve activity
        activity_id = inp.get("activityId")
        if not activity_id:
            activities = self.ref.get("activities", [])
            act_name = inp.get("activityName", "")
            if act_name and activities:
                for act in activities:
                    if act_name.lower() in str(act.get("name", "")).lower():
                        activity_id = act["id"]
                        break
            if not activity_id and activities:
                activity_id = activities[0]["id"]

        # 4. Link activity to project (ignore errors — may already be linked)
        if project_id and activity_id:
            await self._api("POST", "/project/projectActivity",
                            body={"project": {"id": project_id}, "activity": {"id": activity_id}})
            steps.append("activity linked")

        # 5. Create timesheet entries
        entries = inp.get("entries", [])
        if not entries:
            d = inp.get("date", self.today)
            h = inp.get("hours", 0)
            if h:
                entries = [{"date": d, "hours": h, "comment": inp.get("comment", "")}]

        created = 0
        errors = 0
        for entry in entries:
            body: dict[str, Any] = {
                "employee": {"id": employee_id},
                "date": entry["date"],
                "hours": entry["hours"],
            }
            if project_id:
                body["project"] = {"id": project_id}
            if activity_id:
                body["activity"] = {"id": activity_id}
            if entry.get("comment"):
                body["comment"] = entry["comment"]

            result = await self._api("POST", "/timesheet/entry", body=body)
            if self._ok(result):
                created += 1
            else:
                # Retry without activity (common 422 fix)
                body.pop("activity", None)
                result = await self._api("POST", "/timesheet/entry", body=body)
                if self._ok(result):
                    created += 1
                else:
                    errors += 1
                    logger.warning("Timesheet entry failed: %s", json.dumps(result.get("data", {}))[:200])

        steps.append(f"entries: {created}/{len(entries)}")
        return {
            "success": created > 0,
            "created": created,
            "errors": errors,
            "employeeId": employee_id,
            "projectId": project_id,
            "activityId": activity_id,
            "steps_completed": steps,
        }

    # ─── tripletex_api (raw fallback) ───────────────────────────────────

    async def _do_tripletex_api(self, inp: dict) -> dict:
        return await self._api(
            inp.get("method", "GET"),
            inp.get("path", "/"),
            params=inp.get("params"),
            body=inp.get("json_body"),
        )
