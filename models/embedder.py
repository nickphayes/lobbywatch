import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class LobbyingFilingEmbedder:
    """
    Handles embedding generation and similarity search for lobbying filings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence transformer model.

        Args:
            model_name: HuggingFace sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_data = None

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                logging.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logging.error(f"Failed to load model {self.model_name}: {e}")
                # Fallback to a more basic model
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logging.info("Loaded fallback model: all-MiniLM-L6-v2")

    def _create_filing_text(self, filing: Dict) -> str:
        """
        Create a comprehensive text representation of a filing for embedding.

        Args:
            filing: Filing dictionary

        Returns:
            Combined text representation of the filing
        """
        text_parts = []

        # Client information
        client = filing.get("client", {})
        if client:
            client_name = client.get("name", "")
            client_desc = client.get("general_description", "")
            client_state = client.get("state_display", "")

            if client_name:
                text_parts.append(f"Client: {client_name}")
            if client_desc:
                text_parts.append(f"Client Description: {client_desc}")
            if client_state:
                text_parts.append(f"Client Location: {client_state}")

        # Registrant information
        registrant = filing.get("registrant", {})
        if registrant:
            registrant_name = registrant.get("name", "")
            registrant_desc = registrant.get("description", "")
            registrant_state = registrant.get("state_display", "")

            if registrant_name:
                text_parts.append(f"Registrant: {registrant_name}")
            if registrant_desc:
                text_parts.append(f"Registrant Description: {registrant_desc}")
            if registrant_state:
                text_parts.append(f"Registrant Location: {registrant_state}")

        # Filing type and period
        filing_type = filing.get("filing_type_display", "")
        filing_period = filing.get("filing_period_display", "")
        filing_year = filing.get("filing_year", "")

        if filing_type:
            text_parts.append(f"Filing Type: {filing_type}")
        if filing_period and filing_year:
            text_parts.append(f"Period: {filing_period} {filing_year}")

        # Lobbying activities
        activities = filing.get("lobbying_activities", [])
        if activities:
            # Collect all issue codes and descriptions
            issue_codes = []
            descriptions = []

            for activity in activities:
                issue_code = activity.get("general_issue_code_display", "")
                description = activity.get("description", "")

                if issue_code and issue_code not in issue_codes:
                    issue_codes.append(issue_code)
                if description and description.strip():
                    descriptions.append(description.strip())

            if issue_codes:
                text_parts.append(f"Issue Areas: {', '.join(issue_codes)}")

            if descriptions:
                # Combine all activity descriptions
                combined_descriptions = " ".join(descriptions)
                text_parts.append(f"Activities: {combined_descriptions}")

        # Additional metadata
        income = filing.get("income")
        if income:
            text_parts.append(f"Income: {income}")

        # Foreign entities if any
        foreign_entities = filing.get("foreign_entities", [])
        if foreign_entities:
            foreign_names = [
                entity.get("name", "")
                for entity in foreign_entities
                if entity.get("name")
            ]
            if foreign_names:
                text_parts.append(f"Foreign Entities: {', '.join(foreign_names)}")

        # Affiliated organizations if any
        affiliated_orgs = filing.get("affiliated_organizations", [])
        if affiliated_orgs:
            org_names = [
                org.get("name", "") for org in affiliated_orgs if org.get("name")
            ]
            if org_names:
                text_parts.append(f"Affiliated Organizations: {', '.join(org_names)}")

        return " | ".join(text_parts)

    def _create_filing_metadata(self, filing: Dict, filing_idx: int) -> Dict:
        """Create metadata for a filing."""
        client = filing.get("client", {})
        registrant = filing.get("registrant", {})

        # Get all issue codes from activities
        activities = filing.get("lobbying_activities", [])
        issue_codes = []
        for activity in activities:
            issue_code = activity.get("general_issue_code_display", "")
            if issue_code and issue_code not in issue_codes:
                issue_codes.append(issue_code)

        return {
            "filing_idx": filing_idx,
            "filing_uuid": filing.get("filing_uuid", ""),
            "filing_id": str(uuid.uuid4()),
            "client_name": client.get("name", "Unknown Client"),
            "registrant_name": registrant.get("name", "Unknown Registrant"),
            "filing_type": filing.get("filing_type_display", ""),
            "filing_year": filing.get("filing_year", ""),
            "filing_period": filing.get("filing_period_display", ""),
            "issue_codes": issue_codes,
            "activity_count": len(activities),
            "income": filing.get("income"),
            "filing_shorthand": self._create_filing_shorthand(filing),
            "has_foreign_entities": len(filing.get("foreign_entities", [])) > 0,
            "has_affiliated_orgs": len(filing.get("affiliated_organizations", [])) > 0,
        }

    def generate_embeddings_for_filings(
        self, filings: List[Dict], output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate embeddings for all filings in the dataset.

        Args:
            filings: List of filing dictionaries
            output_path: Optional path to save the enhanced data

        Returns:
            Enhanced filings with embeddings added
        """
        self._load_model()

        logging.info(f"Generating embeddings for {len(filings)} filings...")

        # Create text representations for all filings
        filing_texts = []
        filing_metadata = []

        for filing_idx, filing in enumerate(filings):
            filing_text = self._create_filing_text(filing)

            # Only process filings with meaningful content
            if filing_text and len(filing_text.strip()) > 20:
                filing_texts.append(filing_text)
                metadata = self._create_filing_metadata(filing, filing_idx)
                filing_metadata.append(metadata)
            else:
                logging.warning(f"Skipping filing {filing_idx} - insufficient content")

        logging.info(f"Processing {len(filing_texts)} filings with sufficient content")

        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(filing_texts), batch_size):
            batch = filing_texts[i : i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=False)
            all_embeddings.extend(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                logging.info(
                    f"Processed {i + len(batch)} / {len(filing_texts)} filings"
                )

        # Add embeddings back to the original filings
        enhanced_filings = []
        embedding_idx = 0

        for filing_idx, filing in enumerate(filings):
            enhanced_filing = filing.copy()

            # Check if this filing has an embedding
            if embedding_idx < len(filing_metadata):
                metadata = filing_metadata[embedding_idx]
                if metadata["filing_idx"] == filing_idx:
                    enhanced_filing["filing_embedding"] = all_embeddings[
                        embedding_idx
                    ].tolist()
                    enhanced_filing["filing_id"] = metadata["filing_id"]
                    enhanced_filing["filing_text"] = filing_texts[embedding_idx]
                    embedding_idx += 1

            enhanced_filings.append(enhanced_filing)

        # Save enhanced data if output path provided
        if output_path:
            self._save_enhanced_data(enhanced_filings, output_path)

        # Store for similarity search
        self.embeddings_data = {
            "filings": enhanced_filings,
            "metadata": filing_metadata,
            "embeddings": all_embeddings,
            "texts": filing_texts,
            "generated_at": datetime.now().isoformat(),
        }

        logging.info("Filing embedding generation completed successfully")
        return enhanced_filings

    def _create_filing_shorthand(self, filing: Dict) -> str:
        """Create a shorthand identifier for a filing."""
        client = filing.get("client", {})
        registrant = filing.get("registrant", {})

        client_name = client.get("name", "Unknown Client")
        registrant_name = registrant.get("name", "Unknown Registrant")
        filing_year = filing.get("filing_year", "Unknown Year")
        filing_period = filing.get("filing_period_display", "Unknown Period")

        client_short = (
            client_name[:30] + "..." if len(client_name) > 30 else client_name
        )
        registrant_short = (
            registrant_name[:25] + "..."
            if len(registrant_name) > 25
            else registrant_name
        )

        return f"{client_short} | {registrant_short} | {filing_year} {filing_period}"

    def _save_enhanced_data(self, enhanced_filings: List[Dict], output_path: str):
        """Save enhanced filings data to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(enhanced_filings, f, indent=2)
            logging.info(f"Enhanced data saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save enhanced data: {e}")

    def load_embeddings_data(self, enhanced_filings_path: str) -> bool:
        """
        Load pre-computed embeddings data for similarity search.

        Args:
            enhanced_filings_path: Path to the enhanced filings JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(enhanced_filings_path, "r") as f:
                enhanced_filings = json.load(f)

            # Extract embeddings and metadata
            embeddings = []
            metadata = []
            texts = []

            for filing_idx, filing in enumerate(enhanced_filings):
                embedding = filing.get("filing_embedding")
                if embedding:
                    embeddings.append(np.array(embedding))
                    texts.append(filing.get("filing_text", ""))

                    # Create metadata
                    client = filing.get("client", {})
                    registrant = filing.get("registrant", {})
                    activities = filing.get("lobbying_activities", [])

                    issue_codes = []
                    for activity in activities:
                        issue_code = activity.get("general_issue_code_display", "")
                        if issue_code and issue_code not in issue_codes:
                            issue_codes.append(issue_code)

                    metadata.append(
                        {
                            "filing_idx": filing_idx,
                            "filing_uuid": filing.get("filing_uuid", ""),
                            "filing_id": filing.get("filing_id", ""),
                            "client_name": client.get("name", "Unknown Client"),
                            "registrant_name": registrant.get(
                                "name", "Unknown Registrant"
                            ),
                            "filing_type": filing.get("filing_type_display", ""),
                            "filing_year": filing.get("filing_year", ""),
                            "filing_period": filing.get("filing_period_display", ""),
                            "issue_codes": issue_codes,
                            "activity_count": len(activities),
                            "income": filing.get("income"),
                            "filing_shorthand": self._create_filing_shorthand(filing),
                            "has_foreign_entities": len(
                                filing.get("foreign_entities", [])
                            )
                            > 0,
                            "has_affiliated_orgs": len(
                                filing.get("affiliated_organizations", [])
                            )
                            > 0,
                        }
                    )

            self.embeddings_data = {
                "filings": enhanced_filings,
                "metadata": metadata,
                "embeddings": embeddings,
                "texts": texts,
            }

            logging.info(
                f"Loaded {len(embeddings)} filing embeddings from {enhanced_filings_path}"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to load embeddings data: {e}")
            return False

    def find_similar_filings(
        self,
        target_filing: Dict,
        target_filing_idx: int = None,
        top_k: int = 10,
        similarity_threshold: float = 0.6,
        filter_by_issue_code: Optional[str] = None,
        filter_by_year: Optional[int] = None,
    ) -> List[Dict]:
        """
        Find filings similar to the target filing.

        Args:
            target_filing: The filing to find similarities for
            target_filing_idx: Index of the target filing (to exclude from results)
            top_k: Number of similar filings to return
            similarity_threshold: Minimum similarity score to include
            filter_by_issue_code: Optional issue code to filter by
            filter_by_year: Optional year to filter by

        Returns:
            List of similar filings with similarity scores
        """
        if not self.embeddings_data:
            logging.error(
                "No embeddings data loaded. Run generate_embeddings_for_filings() first."
            )
            return []

        target_embedding = target_filing.get("filing_embedding")
        if not target_embedding:
            logging.error("Target filing has no embedding")
            return []

        target_embedding = np.array(target_embedding)
        similarities = []

        # Calculate cosine similarity with all filings
        for idx, (embedding, metadata) in enumerate(
            zip(self.embeddings_data["embeddings"], self.embeddings_data["metadata"])
        ):
            # Skip the target filing itself
            if (
                target_filing_idx is not None
                and metadata["filing_idx"] == target_filing_idx
            ):
                continue

            # Skip if it's the exact same filing
            if target_filing.get("filing_id") and metadata.get(
                "filing_id"
            ) == target_filing.get("filing_id"):
                continue

            # Apply filters
            if filter_by_issue_code:
                if filter_by_issue_code not in metadata.get("issue_codes", []):
                    continue

            if filter_by_year:
                if metadata.get("filing_year") != filter_by_year:
                    continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(target_embedding, embedding)

            if similarity >= similarity_threshold:
                similarities.append(
                    {
                        "similarity_score": float(similarity),
                        "client_name": metadata["client_name"],
                        "registrant_name": metadata["registrant_name"],
                        "filing_type": metadata["filing_type"],
                        "filing_year": metadata["filing_year"],
                        "filing_period": metadata["filing_period"],
                        "issue_codes": metadata["issue_codes"],
                        "activity_count": metadata["activity_count"],
                        "income": metadata["income"],
                        "filing_shorthand": metadata["filing_shorthand"],
                        "filing_idx": metadata["filing_idx"],
                        "filing_uuid": metadata["filing_uuid"],
                        "filing_id": metadata.get("filing_id"),
                        "has_foreign_entities": metadata["has_foreign_entities"],
                        "has_affiliated_orgs": metadata["has_affiliated_orgs"],
                    }
                )

        # Sort by similarity score (descending) and return top-k
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:top_k]

    def search_filings_by_text(
        self,
        search_text: str,
        top_k: int = 10,
        similarity_threshold: float = 0.6,
        filter_by_issue_code: Optional[str] = None,
        filter_by_year: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search for filings similar to the given text query.

        Args:
            search_text: Text to search for similar filings
            top_k: Number of similar filings to return
            similarity_threshold: Minimum similarity score to include
            filter_by_issue_code: Optional issue code to filter by
            filter_by_year: Optional year to filter by

        Returns:
            List of similar filings with similarity scores
        """
        if not self.embeddings_data:
            logging.error(
                "No embeddings data loaded. Run generate_embeddings_for_filings() first."
            )
            return []

        self._load_model()

        # Generate embedding for search text
        search_embedding = self.model.encode([search_text], convert_to_tensor=False)[0]
        similarities = []

        # Calculate cosine similarity with all filings
        for idx, (embedding, metadata) in enumerate(
            zip(self.embeddings_data["embeddings"], self.embeddings_data["metadata"])
        ):
            # Apply filters
            if filter_by_issue_code:
                if filter_by_issue_code not in metadata.get("issue_codes", []):
                    continue

            if filter_by_year:
                if metadata.get("filing_year") != filter_by_year:
                    continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(search_embedding, embedding)

            if similarity >= similarity_threshold:
                similarities.append(
                    {
                        "similarity_score": float(similarity),
                        "client_name": metadata["client_name"],
                        "registrant_name": metadata["registrant_name"],
                        "filing_type": metadata["filing_type"],
                        "filing_year": metadata["filing_year"],
                        "filing_period": metadata["filing_period"],
                        "issue_codes": metadata["issue_codes"],
                        "activity_count": metadata["activity_count"],
                        "income": metadata["income"],
                        "filing_shorthand": metadata["filing_shorthand"],
                        "filing_idx": metadata["filing_idx"],
                        "filing_uuid": metadata["filing_uuid"],
                        "filing_id": metadata.get("filing_id"),
                        "has_foreign_entities": metadata["has_foreign_entities"],
                        "has_affiliated_orgs": metadata["has_affiliated_orgs"],
                    }
                )

        # Sort by similarity score (descending) and return top-k
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_embedding_stats(self) -> Dict:
        """Get statistics about the loaded embeddings."""
        if not self.embeddings_data:
            return {}

        embeddings = self.embeddings_data["embeddings"]
        metadata = self.embeddings_data["metadata"]

        # Collect all issue codes
        all_issue_codes = []
        for m in metadata:
            all_issue_codes.extend(m.get("issue_codes", []))

        unique_issues = set(all_issue_codes)

        # Get year distribution
        years = [m.get("filing_year") for m in metadata if m.get("filing_year")]
        unique_years = set(years)

        # Get client and registrant counts
        clients = set(m.get("client_name") for m in metadata if m.get("client_name"))
        registrants = set(
            m.get("registrant_name") for m in metadata if m.get("registrant_name")
        )

        return {
            "total_filings": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "unique_issue_codes": len(unique_issues),
            "unique_years": len(unique_years),
            "unique_clients": len(clients),
            "unique_registrants": len(registrants),
            "issue_code_distribution": {
                issue: all_issue_codes.count(issue) for issue in unique_issues
            },
            "year_distribution": {year: years.count(year) for year in unique_years},
            "model_name": self.model_name,
            "filings_with_foreign_entities": sum(
                1 for m in metadata if m.get("has_foreign_entities")
            ),
            "filings_with_affiliated_orgs": sum(
                1 for m in metadata if m.get("has_affiliated_orgs")
            ),
        }


# Standalone preprocessing script functionality
def preprocess_filings(
    input_path: str,
    output_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Standalone function to preprocess filings and generate embeddings.

    Args:
        input_path: Path to original filings.json
        output_path: Path to save enhanced filings with embeddings
        model_name: Sentence transformer model to use
    """
    logging.basicConfig(level=logging.INFO)

    try:
        # Load original filings
        with open(input_path, "r") as f:
            filings = json.load(f)

        # Generate embeddings
        embedder = LobbyingFilingEmbedder(model_name)
        enhanced_filings = embedder.generate_embeddings_for_filings(
            filings, output_path
        )

        # Print stats
        stats = embedder.get_embedding_stats()
        print("\nFiling embedding generation completed!")
        print(f"Total filings processed: {stats['total_filings']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Unique issue codes: {stats['unique_issue_codes']}")
        print(f"Unique clients: {stats['unique_clients']}")
        print(f"Unique registrants: {stats['unique_registrants']}")
        print(f"Enhanced data saved to: {output_path}")

        return True

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/filings.json"
    output_file = "data/processed/filings_with_embeddings.json"

    # Run preprocessing
    success = preprocess_filings(input_file, output_file)

    if success:
        # Test similarity search
        embedder = LobbyingFilingEmbedder()
        embedder.load_embeddings_data(output_file)

        # Get stats
        stats = embedder.get_embedding_stats()
        print(f"\nLoaded embeddings stats: {stats}")

        # Example search by text
        search_results = embedder.search_filings_by_text(
            "emergency disaster preparedness funding", top_k=5, similarity_threshold=0.5
        )

        print(f"\nExample search results: {len(search_results)} filings found")
        for result in search_results[:3]:
            print(
                f"- {result['filing_shorthand']} (Score: {result['similarity_score']:.3f})"
            )
