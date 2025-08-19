## Setup Instructions
Follow these steps to set up and run the project locally:

Clone the Repository
- git clone git@github.com:Brandonawan/claim-resubmission-pipeline.git
- cd claim-resubmission-pipeline.git

Create and Activate Virtual Environment
- python -m venv env
- source env/bin/activate  # on linux

## Install Dependencies
- Ensure Python 3.10+ is installed, then:
- pip install -r requirements.txt

## Run commands
- python3 claim_pipeline.py

## Terminal Output
- The script will output the processing status of each claim in the terminal.
- [Output Example](images/claim_pipeline.png)

## outputed files
- The script processes the claims data and generates two JSON files in the current directory.
- The script will generate a json file named `rejection_log.json` in the current directory.
You can view it here: [rejection_log.json](./rejection_log.json)

```json
[
  {
    "record": {
      "claim_id": "A125",
      "patient_id": null,
      "procedure_code": "99215",
      "denial_reason": "Authorization expired",
      "status": "denied",
      "submitted_at": "2025-07-05T00:00:00",
      "source_system": "alpha"
    },
    "error": "Ineligible: Missing patient_id"
  },
  {
    "record": {
      "claim_id": "A126",
      "patient_id": "P003",
      "procedure_code": "99381",
      "denial_reason": null,
      "status": "approved",
      "submitted_at": "2025-07-15T00:00:00",
      "source_system": "alpha"
    },
    "error": "Ineligible: Not denied"
  },
  {
    "record": {
      "claim_id": "B987",
      "patient_id": "P010",
      "procedure_code": "99213",
      "denial_reason": "Incorrect provider type",
      "status": "denied",
      "submitted_at": "2025-07-03T00:00:00",
      "source_system": "beta"
    },
    "error": "Ineligible: Incorrect provider type"
  },
  {
    "record": {
      "claim_id": "B989",
      "patient_id": "P012",
      "procedure_code": "99215",
      "denial_reason": null,
      "status": "approved",
      "submitted_at": "2025-07-10T00:00:00",
      "source_system": "beta"
    },
    "error": "Ineligible: Not denied"
  },
  {
    "record": {
      "claim_id": "B990",
      "patient_id": null,
      "procedure_code": "99401",
      "denial_reason": "incorrect procedure",
      "status": "denied",
      "submitted_at": "2025-07-01T00:00:00",
      "source_system": "beta"
    },
    "error": "Ineligible: Missing patient_id"
  }
]
```

- The script also generates another json file named `resubmission_candidates.json` in the current directory.
You can view it here: [resubmission_candidates.json](./resubmission_candidates.json)

```json
[
  {
    "claim_id": "A123",
    "resubmission_reason": "Missing modifier",
    "source_system": "alpha",
    "recommended_changes": "Review Missing modifier and resubmit"
  },
  {
    "claim_id": "A124",
    "resubmission_reason": "Incorrect NPI",
    "source_system": "alpha",
    "recommended_changes": "Review Incorrect NPI and resubmit"
  },
  {
    "claim_id": "A127",
    "resubmission_reason": "Prior auth required",
    "source_system": "alpha",
    "recommended_changes": "Review Prior auth required and resubmit"
  },
  {
    "claim_id": "B988",
    "resubmission_reason": "Missing modifier",
    "source_system": "beta",
    "recommended_changes": "Review Missing modifier and resubmit"
  }
]
```

