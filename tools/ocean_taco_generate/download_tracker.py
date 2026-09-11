#!/usr/bin/env python3
"""Download tracking and reporting utilities for OceanTACO source downloads."""

import json
import logging
from datetime import datetime
from pathlib import Path


class DownloadTracker:
    """Track download attempts, successes, and failures for reproducibility."""

    def __init__(self, log_dir: Path):
        """Initialize tracker state and configure file/console logging."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = (
            self.log_dir
            / f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        self.results = {
            "downloads": [],
            "errors": [],
            "summary": {},
        }

    def log_download_attempt(
        self, dataset: str, date_range: tuple, status: str, details: dict = None
    ):
        """Log a download attempt."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "date_min": date_range[0],
            "date_max": date_range[1],
            "status": status,
            "details": details or {},
        }
        self.results["downloads"].append(record)

        if status == "failed":
            self.logger.error(
                f"{dataset} [{date_range[0]} to {date_range[1]}]: FAILED - {details}"
            )
        elif status == "partial":
            self.logger.warning(
                f"{dataset} [{date_range[0]} to {date_range[1]}]: PARTIAL - {details}"
            )
        else:
            self.logger.info(
                f"{dataset} [{date_range[0]} to {date_range[1]}]: {status.upper()}"
            )

    def log_error(self, dataset: str, error: Exception, context: dict = None):
        """Log an error with full traceback."""
        import traceback

        error_record = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }
        self.results["errors"].append(error_record)
        self.logger.error(f"{dataset} ERROR: {error}", exc_info=True)

    def save_report(self):
        """Save detailed JSON report of all download attempts."""
        report_file = (
            self.log_dir
            / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        self.results["summary"] = self._generate_summary()

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Download report saved: {report_file}")
        return report_file

    def _generate_summary(self):
        """Generate summary statistics."""
        downloads = self.results["downloads"]

        summary = {
            "total_attempts": len(downloads),
            "successful": len([d for d in downloads if d["status"] == "success"]),
            "partial": len([d for d in downloads if d["status"] == "partial"]),
            "failed": len([d for d in downloads if d["status"] == "failed"]),
            "skipped": len([d for d in downloads if d["status"] == "skipped"]),
            "total_errors": len(self.results["errors"]),
            "datasets": {},
        }

        for download in downloads:
            dataset = download["dataset"]
            if dataset not in summary["datasets"]:
                summary["datasets"][dataset] = {
                    "attempts": 0,
                    "success": 0,
                    "failed": 0,
                }
            summary["datasets"][dataset]["attempts"] += 1
            if download["status"] == "success":
                summary["datasets"][dataset]["success"] += 1
            elif download["status"] == "failed":
                summary["datasets"][dataset]["failed"] += 1

        return summary

    def print_summary(self):
        """Print summary to console."""
        summary = self._generate_summary()

        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"Total attempts: {summary['total_attempts']}")
        print(f"  Successful:  {summary['successful']}")
        print(f"  Partial:     {summary['partial']}")
        print(f"  Failed:      {summary['failed']}")
        print(f"  Skipped:     {summary['skipped']}")
        print(f"  Errors:      {summary['total_errors']}")
        print("\nPer-dataset summary:")
        for dataset, stats in summary["datasets"].items():
            print(f"  {dataset:20s}: {stats['success']}/{stats['attempts']} successful")
        print("=" * 80)
