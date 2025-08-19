import pandas as pd
import json
import logging
from datetime import datetime
from dateutil.parser import parse
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaimProcessor:
    """
    Pipeline to ingest, normalize, and evaluate insurance claims for resubmission eligibility.
    Handles multiple EMR sources, applies business rules, and outputs candidates.
    """
    def __init__(self, current_date="2025-07-30"):
        self.current_date = parse(current_date).date()
        self.retryable_reasons = {"Missing modifier", "Incorrect NPI", "Prior auth required"}
        self.non_retryable_reasons = {"Authorization expired", "Incorrect provider type"}
        self.normalized_claims = []
        self.eligible_claims = []
        self.rejected_records = []
        self.metrics = {"total_claims": 0, "alpha_claims": 0, "beta_claims": 0, "eligible": 0, "rejected": 0}

    def load_alpha_csv(self, file_path: str) -> list:
        """Load and normalize emr_alpha.csv data."""
        logger.info(f"Loading CSV from {file_path}")
        try:
            df = pd.read_csv(file_path)
            self.metrics["alpha_claims"] = len(df)
            claims = []
            for _, row in df.iterrows():
                try:
                    claim = {
                        "claim_id": str(row["claim_id"]).strip(),
                        "patient_id": str(row["patient_id"]).strip() if pd.notna(row["patient_id"]) else None,
                        "procedure_code": str(row["procedure_code"]).strip(),
                        "denial_reason": str(row["denial_reason"]).strip() if pd.notna(row["denial_reason"]) else None,
                        "status": str(row["status"]).strip().lower(),
                        "submitted_at": self.normalize_date(row["submitted_at"]),
                        "source_system": "alpha"
                    }
                    claims.append(claim)
                except Exception as e:
                    logger.error(f"Error processing alpha record {row.get('claim_id', 'unknown')}: {str(e)}")
                    self.rejected_records.append({"record": row.to_dict(), "error": str(e)})
            return claims
        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
            return []
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
            return []

    def load_beta_json(self, file_path: str) -> list:
        """Load and normalize emr_beta.json data."""
        logger.info(f"Loading JSON from {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.metrics["beta_claims"] = len(data)
            claims = []
            for record in data:
                try:
                    claim = {
                        "claim_id": str(record["id"]).strip(),
                        "patient_id": str(record["member"]).strip() if record.get("member") else None,
                        "procedure_code": str(record["code"]).strip(),
                        "denial_reason": str(record["error_msg"]).strip() if record.get("error_msg") else None,
                        "status": str(record["status"]).strip().lower(),
                        "submitted_at": self.normalize_date(record["date"]),
                        "source_system": "beta"
                    }
                    claims.append(claim)
                except Exception as e:
                    logger.error(f"Error processing beta record {record.get('id', 'unknown')}: {str(e)}")
                    self.rejected_records.append({"record": record, "error": str(e)})
            return claims
        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
            return []
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {str(e)}")
            return []

    def normalize_date(self, date_str: str) -> str:
        """Convert date to ISO format (YYYY-MM-DDTHH:MM:SS)."""
        try:
            parsed_date = parse(date_str)
            return parsed_date.isoformat()
        except Exception as e:
            logger.warning(f"Invalid date format {date_str}: {str(e)}. Using null.")
            return None

    def is_retryable_reason(self, reason: str) -> bool:
        """Determine if denial reason is retryable, using mocked LLM for ambiguous cases."""
        if reason in self.retryable_reasons:
            return True
        if reason in self.non_retryable_reasons:
            return False
        # Mocked LLM classifier for ambiguous reasons
        if reason == "incorrect procedure":
            logger.info(f"Mocked LLM classified '{reason}' as retryable")
            return True
        logger.info(f"Mocked LLM classified '{reason}' as non-retryable")
        return False

    def is_eligible_for_resubmission(self, claim: dict) -> tuple[bool, str]:
        """Check if claim is eligible for resubmission."""
        if claim["status"] != "denied":
            return False, "Not denied"
        if not claim["patient_id"]:
            return False, "Missing patient_id"
        try:
            submitted_date = parse(claim["submitted_at"]).date()
            days_since_submission = (self.current_date - submitted_date).days
            if days_since_submission <= 7:
                return False, "Submitted within 7 days"
        except Exception:
            return False, "Invalid submission date"
        if not claim["denial_reason"]:
            return self.is_retryable_reason(None), "Null denial reason"
        return self.is_retryable_reason(claim["denial_reason"]), claim["denial_reason"]

    def process_claims(self, alpha_file: str, beta_file: str):
        """Main pipeline to process claims from both sources."""
        logger.info("Starting claim processing pipeline")
        # Load and normalize data
        self.normalized_claims.extend(self.load_alpha_csv(alpha_file))
        self.normalized_claims.extend(self.load_beta_json(beta_file))
        self.metrics["total_claims"] = len(self.normalized_claims)

        # Evaluate eligibility
        for claim in self.normalized_claims:
            is_eligible, reason = self.is_eligible_for_resubmission(claim)
            if is_eligible:
                self.eligible_claims.append({
                    "claim_id": claim["claim_id"],
                    "resubmission_reason": claim["denial_reason"],
                    "source_system": claim["source_system"],
                    "recommended_changes": f"Review {claim['denial_reason']} and resubmit" if claim["denial_reason"] else "Review claim details and resubmit"
                })
                self.metrics["eligible"] += 1
            else:
                self.rejected_records.append({"record": claim, "error": f"Ineligible: {reason}"})
                self.metrics["rejected"] += 1

    def save_output(self, output_file: str, rejection_file: str):
        """Save eligible claims and rejected records to JSON files."""
        logger.info(f"Saving eligible claims to {output_file}")
        try:
            with open(output_file, 'w') as f:
                json.dump(self.eligible_claims, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {str(e)}")

        logger.info(f"Saving rejected records to {rejection_file}")
        try:
            with open(rejection_file, 'w') as f:
                json.dump(self.rejected_records, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving to {rejection_file}: {str(e)}")

    def log_metrics(self):
        """Log pipeline metrics."""
        logger.info("Pipeline metrics:")
        for key, value in self.metrics.items():
            logger.info(f"{key}: {value}")

def main():
    processor = ClaimProcessor(current_date="2025-07-30")
    processor.process_claims("emr_alpha.csv", "emr_beta.json")
    processor.save_output("resubmission_candidates.json", "rejection_log.json")
    processor.log_metrics()

if __name__ == "__main__":
    main()