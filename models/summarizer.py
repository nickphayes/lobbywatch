import logging
from typing import Dict, List, Union

from transformers import pipeline


class LobbyingSummarizer:
    """summarizer for lobbying disclosure data with customizable prompts and multiple summary types."""

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the summarizer with a specified model.

        Args:
            model_name: HuggingFace model name. Alternatives:
                - "facebook/bart-large-cnn" (default, good balance)
                - "sshleifer/distilbart-cnn-12-6" (smaller, faster)
                - "google/pegasus-xsum" (more abstractive)
        """
        try:
            self.summarizer = pipeline("summarization", model=model_name)
            self.model_name = model_name
        except Exception:
            logging.warning(
                f"Failed to load {model_name}, falling back to bart-large-cnn"
            )
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            self.model_name = "facebook/bart-large-cnn"

    def _prepare_text_with_context(
        self, text: str, context_type: str = "general"
    ) -> str:
        """
        context-specific prefixes to guide the summarization.

        Args:
            text: The text to summarize
            context_type: Type of summary context ('activity', 'client', 'registrant', 'general')
        """
        context_prefixes = {
            "activity": "This lobbying activity involves: ",
            "client": "This client organization: ",
            "registrant": "This lobbying firm: ",
            "filing": "This lobbying disclosure filing: ",
            "general": "",
        }

        prefix = context_prefixes.get(context_type, "")
        return f"{prefix}{text}"

    def summarize_activity(
        self,
        description: str,
        max_tokens: int = 150,
        min_words: int = 20,
        context_aware: bool = True,
    ) -> str:
        """
        Summarize a lobbying activity description.

        Args:
            description: The activity description to summarize
            max_tokens: Maximum length of summary
            min_words: Minimum words required to attempt summarization
            context_aware: Whether to add context-specific guidance
        """
        if not description or len(description.split()) < min_words:
            return description

        text = (
            self._prepare_text_with_context(description, "activity")
            if context_aware
            else description
        )

        try:
            summary = self.summarizer(
                text,
                max_length=min(max_tokens, 150),  # Cap at model limits
                min_length=min(20, max_tokens // 3),
                do_sample=False,
                truncation=True,
            )
            return summary[0]["summary_text"]
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return (
                description[:max_tokens] + "..."
                if len(description) > max_tokens
                else description
            )

    def summarize_filing(
        self,
        filing_data: Dict,
        include_activities: bool = True,
        include_financials: bool = True,
    ) -> Dict[str, str]:
        """
        Create a comprehensive summary of an entire lobbying filing.

        Args:
            filing_data: The complete filing data dictionary
            include_activities: Whether to include activity summaries
            include_financials: Whether to include financial information
        """
        summary = {}

        # Basic filing info
        client_name = filing_data.get("client", {}).get("name", "Unknown Client")
        registrant_name = filing_data.get("registrant", {}).get(
            "name", "Unknown Registrant"
        )
        filing_period = filing_data.get("filing_period_display", "Unknown Period")
        filing_year = filing_data.get("filing_year", "Unknown Year")

        summary["overview"] = (
            f"{registrant_name} lobbied for {client_name} during {filing_period} {filing_year}."
        )

        # Client summary
        if "client" in filing_data:
            client_desc = filing_data["client"].get("general_description", "")
            if client_desc:
                summary["client"] = self._prepare_text_with_context(
                    f"{client_name} is {client_desc}", "client"
                )

        # Activity summaries
        if include_activities and "lobbying_activities" in filing_data:
            activities = filing_data["lobbying_activities"]
            if activities:
                activity_summaries = []
                issue_codes = set()

                for activity in activities:
                    issue_code = activity.get(
                        "general_issue_code_display", "Unknown Issue"
                    )
                    issue_codes.add(issue_code)

                    desc = activity.get("description", "")
                    if desc:
                        activity_summary = self.summarize_activity(desc, max_tokens=100)
                        activity_summaries.append(f"• {issue_code}: {activity_summary}")

                summary["activities"] = "\n".join(activity_summaries)
                summary["issue_areas"] = (
                    f"Issue areas: {', '.join(sorted(issue_codes))}"
                )

        # Financial info
        if include_financials:
            income = filing_data.get("income")
            expenses = filing_data.get("expenses")
            if income:
                summary["financials"] = f"Reported income: ${income:,}"
            elif expenses:
                summary["financials"] = f"Reported expenses: ${expenses:,}"

        # Lobbyist info
        lobbyists = self._extract_unique_lobbyists(filing_data)
        if lobbyists:
            summary["lobbyists"] = f"Lobbyists: {', '.join(lobbyists)}"

        return summary

    def _extract_unique_lobbyists(self, filing_data: Dict) -> List[str]:
        """Extract unique lobbyist names from filing data."""
        lobbyists = set()
        for activity in filing_data.get("lobbying_activities", []):
            for lobbyist_info in activity.get("lobbyists", []):
                lobbyist = lobbyist_info.get("lobbyist", {})
                first_name = lobbyist.get("first_name", "")
                last_name = lobbyist.get("last_name", "")
                if first_name and last_name:
                    lobbyists.add(f"{first_name} {last_name}")
        return list(lobbyists)

    def batch_summarize_activities(
        self, activities: List[str], max_workers: int = 4
    ) -> List[str]:
        """
        Summarize multiple activities efficiently.

        Args:
            activities: List of activity descriptions
            max_workers: Number of parallel workers (not used with transformers pipeline)
        """
        return [self.summarize_activity(activity) for activity in activities]

    def get_summary_stats(
        self, original_text: str, summary: str
    ) -> Dict[str, Union[int, float]]:
        """Get statistics about the summarization."""
        return {
            "original_words": len(original_text.split()),
            "summary_words": len(summary.split()),
            "compression_ratio": len(summary.split()) / len(original_text.split())
            if original_text
            else 0,
            "original_chars": len(original_text),
            "summary_chars": len(summary),
        }
