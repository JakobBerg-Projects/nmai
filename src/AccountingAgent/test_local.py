#!/usr/bin/env python3
"""Local testing script for the Tripletex accounting agent.

Usage:
    # Start the server first:
    #   cd src/AccountingAgent && uvicorn main:app --port 8080

    # Then run test scenarios:
    python test_local.py --task employee
    python test_local.py --task customer
    python test_local.py --task invoice
    python test_local.py --task travel
    python test_local.py --task voucher
    python test_local.py --task product
    python test_local.py --task supplier
    python test_local.py --task project
    python test_local.py --all
    python test_local.py --prompt "Opprett en kunde med navn Test AS"
"""

import argparse
import json
import sys
import time

import httpx

AGENT_URL = "http://localhost:8080/solve"

SANDBOX_BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjI5MjA1LCJ0b2tlbiI6ImM5MmI0Yjk2LTVjNTgtNGYyMS1hNjg0LTM0M2Q4ZTRiMjAxYSJ9"

TASK_PROMPTS = {
    "employee": (
        "Opprett en ansatt med følgende informasjon:\n"
        "Fornavn: Ola\n"
        "Etternavn: Nordmann\n"
        "E-post: ola.nordmann@example.com\n"
        "Mobilnummer: 99887766\n"
        "Rollen skal være regnskapsansvarlig."
    ),
    "customer": (
        "Opprett en kunde med følgende informasjon:\n"
        "Navn: Testbedrift AS\n"
        "Organisasjonsnummer: 123456789\n"
        "E-post: post@testbedrift.no\n"
        "Telefon: 22334455\n"
        "Adresse: Storgata 1, 0182 Oslo, Norge"
    ),
    "invoice": (
        "Opprett en faktura for en ny kunde:\n"
        "Kunde: Fakturakunde AS\n"
        "E-post: faktura@kunde.no\n"
        "Ordrelinje 1: Konsulenttjenester, 10 timer à 1500 kr eks. mva (25% mva)\n"
        "Ordrelinje 2: Reiseutgifter, 1 stk à 2000 kr eks. mva (25% mva)\n"
        "Fakturadato: i dag"
    ),
    "travel": (
        "Opprett en reiseregning for en ny ansatt:\n"
        "Ansatt: Kari Reisen\n"
        "Etternavn: Nordmann\n"
        "Tittel: Kundebesøk Bergen\n"
        "Avreise: Oslo, 2026-03-15\n"
        "Ankomst: Bergen\n"
        "Retur: 2026-03-17\n"
        "Utlegg: Hotell, 2500 kr\n"
        "Lever reiseregningen."
    ),
    "voucher": (
        "Opprett et bilag (journal entry) med følgende:\n"
        "Dato: i dag\n"
        "Beskrivelse: Innbetaling fra kunde\n"
        "Debet: Bankkonto (1920), 10000 kr\n"
        "Kredit: Kundefordringer (1500), 10000 kr"
    ),
    "product": (
        "Opprett et produkt:\n"
        "Navn: Konsulenttime\n"
        "Pris eks. mva: 1200 kr\n"
        "MVA-sats: 25%"
    ),
    "supplier": (
        "Opprett en leverandør:\n"
        "Navn: Leveransen AS\n"
        "Organisasjonsnummer: 987654321\n"
        "E-post: kontakt@leveransen.no\n"
        "Telefon: 11223344"
    ),
    "project": (
        "Create a project with the following details:\n"
        "Project name: Website Redesign\n"
        "Start date: 2026-03-01\n"
        "End date: 2026-06-30\n"
        "Create a new employee named Erik Prosjekt as project manager."
    ),
}


def send_task(prompt: str, task_name: str = "test") -> dict:
    payload = {
        "prompt": prompt,
        "files": [],
        "task_id": f"test_{task_name}",
        "tripletex_credentials": {
            "base_url": SANDBOX_BASE_URL,
            "session_token": SANDBOX_TOKEN,
        },
    }

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Prompt: {prompt[:200]}...")
    print(f"{'='*60}")

    start = time.time()
    try:
        resp = httpx.post(
            AGENT_URL,
            json=payload,
            timeout=httpx.Timeout(300.0),
        )
        elapsed = time.time() - start
        print(f"\nStatus: {resp.status_code}")
        print(f"Elapsed: {elapsed:.1f}s")
        try:
            body = resp.json()
            print(f"Response: {json.dumps(body, indent=2, ensure_ascii=False)}")
            return body
        except Exception:
            print(f"Response text: {resp.text[:500]}")
            return {"error": resp.text}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nERROR after {elapsed:.1f}s: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test the accounting agent locally")
    parser.add_argument("--task", choices=list(TASK_PROMPTS.keys()), help="Run a predefined task")
    parser.add_argument("--all", action="store_true", help="Run all predefined tasks")
    parser.add_argument("--prompt", type=str, help="Run a custom prompt")
    parser.add_argument("--url", type=str, default=AGENT_URL, help="Agent URL")
    args = parser.parse_args()

    global AGENT_URL
    AGENT_URL = args.url

    if args.all:
        results = {}
        for name, prompt in TASK_PROMPTS.items():
            results[name] = send_task(prompt, name)
        print(f"\n\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, result in results.items():
            status = "OK" if result.get("status") == "completed" else "MAYBE"
            print(f"  {name}: {status}")
    elif args.task:
        send_task(TASK_PROMPTS[args.task], args.task)
    elif args.prompt:
        send_task(args.prompt, "custom")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
