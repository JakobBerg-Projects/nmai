"""Lightweight task classifier using keyword matching across 7 languages."""

from enum import Enum


class TaskType(Enum):
    CREATE_EMPLOYEE = "create_employee"
    UPDATE_EMPLOYEE = "update_employee"
    CREATE_CUSTOMER = "create_customer"
    CREATE_CONTACT = "create_contact"
    CREATE_SUPPLIER = "create_supplier"
    CREATE_SUPPLIER_INVOICE = "create_supplier_invoice"
    CREATE_PRODUCT = "create_product"
    CREATE_ORDER = "create_order"
    CREATE_INVOICE = "create_invoice"
    REGISTER_PAYMENT = "register_payment"
    CREATE_CREDIT_NOTE = "create_credit_note"
    CREATE_REMINDER = "create_reminder"
    CREATE_PROJECT = "create_project"
    CREATE_TIMESHEET = "create_timesheet"
    CREATE_DEPARTMENT = "create_department"
    CREATE_TRAVEL_EXPENSE = "create_travel_expense"
    CREATE_EMPLOYEE_EXPENSE = "create_employee_expense"
    CREATE_VOUCHER = "create_voucher"
    DELETE_ENTITY = "delete_entity"
    OPENING_BALANCE = "opening_balance"
    YEAR_END_CLOSING = "year_end_closing"
    BANK_RECONCILIATION = "bank_reconciliation"
    UNKNOWN = "unknown"


_KW = {
    "employee": [
        "ansatt", "employee", "mitarbeiter", "employé", "empleado", "funcionário",
        "medarbeider", "personalopplysninger",
    ],
    "customer": [
        "kunde", "customer", "klient", "client", "cliente", "kundeforhold",
    ],
    "supplier": [
        "leverandør", "supplier", "lieferant", "fournisseur", "proveedor", "fornecedor",
    ],
    "product": [
        "produkt", "product", "produit", "producto", "produto", "vare ", "varenummer",
    ],
    "department": [
        "avdeling", "department", "abteilung", "département", "departamento",
    ],
    "project": [
        "prosjekt", "project", "projekt", "projet", "proyecto", "projeto",
    ],
    "invoice": [
        "faktura", "invoice", "rechnung", "facture", "factura", "fatura",
        "fakturere", "fakturering",
    ],
    "travel": [
        "reiseregning", "travel expense", "reisekosten", "note de frais",
        "gasto de viaje", "despesa de viagem", "reiserekning", "tjenestereise",
        "diett", "kilometergodtgjørelse", "kjøregodtgjørelse",
    ],
    "emp_expense": [
        "ansattutlegg", "employee expense", "utlegg", "expense report",
    ],
    "payment": [
        "betaling", "payment", "zahlung", "paiement", "pago", "pagamento",
        "innbetaling", "betalt", "betal ",
    ],
    "credit_note": [
        "kreditnota", "credit note", "gutschrift", "avoir", "nota de crédito",
        "kreditere",
    ],
    "reminder": [
        "purring", "reminder", "mahnung", "rappel", "recordatorio", "lembrete",
        "inkassovarsel", "betalingspåminnelse", "påminnelse",
    ],
    "voucher": [
        "bilag", "voucher", "bokføring", "buchung", "écriture", "asiento",
        "lançamento", "bokfør", "posteringer", "kontering",
    ],
    "timesheet": [
        "timer", "timesheet", "hours", "tid", "stunden", "heures", "horas",
        "timeliste", "timeregistrering", "timeføring",
    ],
    "contact": [
        "kontaktperson", "contact person", "kontakt",
    ],
    "delete": [
        "slett", "fjern", "delete", "remove", "löschen", "supprimer", "eliminar",
        "excluir", "reverser",
    ],
    "update": [
        "oppdater", "endre", "update", "modify", "änder", "modifier", "actualizar",
        "atualizar", "rett opp", "korriger",
    ],
    "opening": [
        "åpningsbalanse", "inngående balanse", "opening balance", "eröffnungsbilanz",
        "ib ", "åpningsbalanser",
    ],
    "yearend": [
        "årsavslutning", "year-end", "year end", "jahresabschluss", "clôture annuelle",
        "årsoppgjør", "årsregnskap",
    ],
    "bank_rec": [
        "bankavsteming", "bank reconciliation", "kontoavstemming", "bankabstimmung",
        "bankavstemming", "bankutskrift",
    ],
    "supplier_inv": [
        "leverandørfaktura", "supplier invoice", "lieferantenrechnung",
        "facture fournisseur", "inngående faktura",
    ],
    "salary": [
        "lønn", "lønnskjøring", "salary", "payroll", "gehalt",
    ],
}


import re

def _strip_noise(text: str) -> str:
    """Remove email addresses, URLs, and org numbers that cause false keyword matches."""
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    return text


def _has(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def classify_task(prompt: str) -> TaskType:
    p = _strip_noise(prompt.lower())

    if _has(p, _KW["opening"]):
        return TaskType.OPENING_BALANCE
    if _has(p, _KW["yearend"]):
        return TaskType.YEAR_END_CLOSING
    if _has(p, _KW["bank_rec"]):
        return TaskType.BANK_RECONCILIATION

    if _has(p, _KW["delete"]):
        if _has(p, _KW["travel"]) or _has(p, _KW["emp_expense"]):
            return TaskType.CREATE_TRAVEL_EXPENSE
        return TaskType.DELETE_ENTITY

    if _has(p, _KW["travel"]):
        return TaskType.CREATE_TRAVEL_EXPENSE
    if _has(p, _KW["emp_expense"]) and not _has(p, _KW["invoice"]):
        return TaskType.CREATE_EMPLOYEE_EXPENSE

    if _has(p, _KW["supplier_inv"]):
        return TaskType.CREATE_SUPPLIER_INVOICE
    if _has(p, _KW["supplier"]) and _has(p, _KW["invoice"]):
        return TaskType.CREATE_SUPPLIER_INVOICE

    if _has(p, _KW["credit_note"]):
        return TaskType.CREATE_CREDIT_NOTE
    if _has(p, _KW["reminder"]):
        return TaskType.CREATE_REMINDER

    if _has(p, _KW["payment"]) and _has(p, _KW["invoice"]):
        return TaskType.REGISTER_PAYMENT

    if _has(p, _KW["invoice"]):
        return TaskType.CREATE_INVOICE

    if _has(p, _KW["timesheet"]):
        return TaskType.CREATE_TIMESHEET
    if _has(p, _KW["project"]):
        return TaskType.CREATE_PROJECT
    if _has(p, _KW["voucher"]):
        return TaskType.CREATE_VOUCHER
    if _has(p, _KW["salary"]):
        return TaskType.CREATE_VOUCHER
    if _has(p, _KW["department"]):
        return TaskType.CREATE_DEPARTMENT
    if _has(p, _KW["contact"]) and _has(p, _KW["customer"]):
        return TaskType.CREATE_CONTACT
    if _has(p, _KW["supplier"]):
        return TaskType.CREATE_SUPPLIER
    if _has(p, _KW["product"]):
        return TaskType.CREATE_PRODUCT
    if _has(p, _KW["customer"]):
        return TaskType.CREATE_CUSTOMER

    if _has(p, _KW["employee"]):
        if _has(p, _KW["update"]):
            return TaskType.UPDATE_EMPLOYEE
        return TaskType.CREATE_EMPLOYEE

    if _has(p, _KW["contact"]):
        return TaskType.CREATE_CONTACT

    return TaskType.UNKNOWN


SECTION_MAP: dict[TaskType, list[str]] = {
    TaskType.CREATE_EMPLOYEE: ["employee"],
    TaskType.UPDATE_EMPLOYEE: ["employee"],
    TaskType.CREATE_CUSTOMER: ["customer", "contact"],
    TaskType.CREATE_CONTACT: ["customer", "contact"],
    TaskType.CREATE_SUPPLIER: ["supplier"],
    TaskType.CREATE_SUPPLIER_INVOICE: ["supplier", "supplier_invoice"],
    TaskType.CREATE_PRODUCT: ["product"],
    TaskType.CREATE_ORDER: ["order", "product", "customer"],
    TaskType.CREATE_INVOICE: ["customer", "order", "invoice", "product", "payment", "contact"],
    TaskType.REGISTER_PAYMENT: ["invoice", "payment"],
    TaskType.CREATE_CREDIT_NOTE: ["invoice", "credit_note"],
    TaskType.CREATE_REMINDER: ["invoice", "reminder"],
    TaskType.CREATE_PROJECT: ["employee", "project", "customer"],
    TaskType.CREATE_TIMESHEET: ["employee", "project", "timesheet"],
    TaskType.CREATE_DEPARTMENT: ["department"],
    TaskType.CREATE_TRAVEL_EXPENSE: ["employee", "travel_expense"],
    TaskType.CREATE_EMPLOYEE_EXPENSE: ["employee", "travel_expense"],
    TaskType.CREATE_VOUCHER: ["voucher"],
    TaskType.DELETE_ENTITY: ["corrections"],
    TaskType.OPENING_BALANCE: ["voucher", "opening_balance"],
    TaskType.YEAR_END_CLOSING: ["voucher", "year_end_closing"],
    TaskType.BANK_RECONCILIATION: ["bank_reconciliation", "voucher"],
    TaskType.UNKNOWN: [],
}

FAST_PATH_ELIGIBLE = {
    TaskType.CREATE_EMPLOYEE,
    TaskType.CREATE_CUSTOMER,
}
