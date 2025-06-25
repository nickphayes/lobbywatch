import json
import os
import random
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

try:
    from embedder import LobbyingFilingEmbedder
    from summarizer import LobbyingSummarizer
except ImportError as e:
    st.error(
        f"Could not import required modules: {e}. Please ensure summarizer.py and embedder.py are in the models/ directory."
    )
    st.stop()

# Page config
st.set_page_config(
    page_title="Lobbying Disclosure Analyzer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Cache functions for better performance
@st.cache_data
def load_filings_data():
    """Load and cache the filings data."""
    try:
        data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "processed",
            "filings_with_embeddings.json",
        )
        with open(data_path, "r") as f:
            filings = json.load(f)
        return filings
    except FileNotFoundError:
        st.error(f"Could not find filings.json at expected path: {data_path}")
        return []
    except json.JSONDecodeError:
        st.error("Error parsing filings.json - please check the file format.")
        return []


@st.cache_resource
def get_summarizer():
    """Initialize and cache the summarizer."""
    return LobbyingSummarizer()


@st.cache_resource
def get_embedder():
    """Initialize and cache the embedder."""
    embedder = LobbyingFilingEmbedder()
    enhanced_data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "processed",
        "filings_with_embeddings.json",
    )

    if os.path.exists(enhanced_data_path):
        success = embedder.load_embeddings_data(enhanced_data_path)
        if success:
            return embedder
    return None


@st.cache_data
def create_filings_dataframe(filings: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame from filings data for table display."""
    data = []

    for idx, filing in enumerate(filings):
        client = filing.get("client", {})
        registrant = filing.get("registrant", {})
        activities = filing.get("lobbying_activities", [])

        # Get issue codes
        issue_codes = []
        for activity in activities:
            issue_code = activity.get("general_issue_code_display", "")
            if issue_code and issue_code not in issue_codes:
                issue_codes.append(issue_code)

        # Get income/expenses and ensure they are numeric
        income = filing.get("income", 0)
        expenses = filing.get("expenses", 0)

        # Convert to float if they are strings, handle None/empty values
        try:
            income = float(income) if income else 0
        except (ValueError, TypeError):
            income = 0

        try:
            expenses = float(expenses) if expenses else 0
        except (ValueError, TypeError):
            expenses = 0

        amount = income if income else expenses

        data.append(
            {
                "Index": idx,
                "Client": client.get("name", "Unknown"),
                "Registrant": registrant.get("name", "Unknown"),
                "Filing Year": filing.get("filing_year", ""),
                "Filing Period": filing.get("filing_period_display", ""),
                "Filing Type": filing.get("filing_type_display", ""),
                "Issue Areas": ", ".join(issue_codes) if issue_codes else "None",
                "Activities": len(activities),
                "Amount": f"${amount:,.0f}" if amount > 0 else "Not reported",
                "Foreign Entities": "Yes" if filing.get("foreign_entities") else "No",
                "Posted Date": filing.get("dt_posted", "")[:10]
                if filing.get("dt_posted")
                else "",
            }
        )

    return pd.DataFrame(data)


def create_filing_shorthand(filing: Dict) -> str:
    """Create a logical shorthand identifier for a filing."""
    client_name = filing.get("client", {}).get("name", "Unknown Client")
    registrant_name = filing.get("registrant", {}).get("name", "Unknown Registrant")
    filing_year = filing.get("filing_year", "Unknown Year")
    filing_period = filing.get("filing_period_display", "Unknown Period")

    client_short = client_name[:30] + "..." if len(client_name) > 30 else client_name
    registrant_short = (
        registrant_name[:25] + "..." if len(registrant_name) > 25 else registrant_name
    )

    shorthand = f"{client_short} | {registrant_short} | {filing_year} {filing_period}"
    return shorthand


def display_summary(summary: Dict[str, str]):
    """Display the AI-generated summary in a structured format."""
    st.markdown("### AI-Generated Summary")
    st.info(
        "The following summary was automatically generated using AI and may not capture all nuances of the original filing."
    )

    if "overview" in summary:
        st.markdown("#### Overview")
        st.write(summary["overview"])

    if "client" in summary:
        st.markdown("#### Client Information")
        st.write(summary["client"])

    if "issue_areas" in summary:
        st.markdown("#### Issue Areas")
        st.write(summary["issue_areas"])

    if "activities" in summary:
        st.markdown("#### Lobbying Activities")
        activities = summary["activities"]

        if isinstance(activities, list):
            for activity in activities:
                if isinstance(activity, dict):
                    with st.container():
                        st.markdown(f"**{activity['issue_code']}**")
                        st.write(activity["summary"])
                        st.markdown("---")
                else:
                    st.markdown(f"• {activity}")
        else:
            activities_list = activities.split("\n")
            for activity in activities_list:
                if activity.strip():
                    st.markdown(activity)

    if "lobbyists" in summary:
        st.markdown("#### Lobbyists")
        st.write(summary["lobbyists"])

    if "financials" in summary:
        st.markdown("#### Financial Information")
        st.write(summary["financials"])


def display_similar_filings(similar_filings: List[Dict], filings: List[Dict]):
    """Display similar filings found by the embedder."""
    st.markdown("### Similar Lobbying Filings")

    if not similar_filings:
        st.info("No similar filings found with the current similarity threshold.")
        return

    st.info(f"Found {len(similar_filings)} similar filings:")

    for i, similar_filing in enumerate(similar_filings):
        with st.expander(
            f"Similar Filing #{i + 1} (Similarity: {similar_filing['similarity_score']:.2%})",
            expanded=i < 3,
        ):
            st.metric("Similarity Score", f"{similar_filing['similarity_score']:.2%}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Client:**")
                st.write(similar_filing["client_name"])

                st.markdown("**Registrant:**")
                st.write(similar_filing["registrant_name"])

                st.markdown("**Filing Period:**")
                st.write(
                    f"{similar_filing['filing_year']} {similar_filing['filing_period']}"
                )

            with col2:
                st.markdown("**Issue Areas:**")
                issue_codes = similar_filing.get("issue_codes", [])
                if issue_codes:
                    for code in issue_codes:
                        st.code(code)
                else:
                    st.write("None specified")

                st.markdown("**Activity Count:**")
                st.write(similar_filing["activity_count"])

                if similar_filing.get("income"):
                    try:
                        income = float(similar_filing["income"])
                        st.markdown("**Income:**")
                        st.write(f"${income:,.0f}")
                    except (ValueError, TypeError):
                        st.markdown("**Income:**")
                        st.write(similar_filing["income"])

            if similar_filing.get("has_foreign_entities"):
                st.info("Has foreign entities")

            if similar_filing.get("has_affiliated_orgs"):
                st.info("Has affiliated organizations")

            # Fixed: Find the actual filing by various identifiers
            filing_idx = None

            # Try multiple methods to find the filing
            if similar_filing.get("filing_id"):
                for idx, filing in enumerate(filings):
                    if filing.get("filing_id") == similar_filing["filing_id"]:
                        filing_idx = idx
                        break

            if filing_idx is None and similar_filing.get("filing_uuid"):
                for idx, filing in enumerate(filings):
                    if filing.get("filing_uuid") == similar_filing["filing_uuid"]:
                        filing_idx = idx
                        break

            if filing_idx is None:
                original_idx = similar_filing.get("filing_idx")
                if original_idx is not None and 0 <= original_idx < len(filings):
                    filing_idx = original_idx

            if filing_idx is not None:
                if st.button(
                    f"View Full Filing #{filing_idx + 1}",
                    key=f"view_filing_{similar_filing.get('filing_id', i)}",
                ):
                    st.session_state["selected_filing_idx"] = filing_idx
                    st.session_state["current_page"] = "analyze"
                    st.rerun()
            else:
                st.warning("Unable to locate this filing in the current dataset")


def display_filing_metadata(filing: Dict):
    """Display key metadata about the filing."""
    st.markdown("### Filing Details")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Filing Year", filing.get("filing_year", "N/A"))
        st.metric("Filing Type", filing.get("filing_type_display", "N/A"))

        income = filing.get("income")
        expenses = filing.get("expenses")

        # Convert to numeric values with proper error handling
        try:
            income_val = float(income) if income else 0
        except (ValueError, TypeError):
            income_val = 0

        try:
            expenses_val = float(expenses) if expenses else 0
        except (ValueError, TypeError):
            expenses_val = 0

        if income_val > 0:
            st.metric("Reported Income", f"${income_val:,.0f}")
        elif expenses_val > 0:
            st.metric("Reported Expenses", f"${expenses_val:,.0f}")

    with col2:
        st.metric("Filing Period", filing.get("filing_period_display", "N/A"))

        num_activities = len(filing.get("lobbying_activities", []))
        st.metric("Lobbying Activities", num_activities)

        unique_lobbyists = set()
        for activity in filing.get("lobbying_activities", []):
            for lobbyist_info in activity.get("lobbyists", []):
                lobbyist = lobbyist_info.get("lobbyist", {})
                first_name = lobbyist.get("first_name", "")
                last_name = lobbyist.get("last_name", "")
                if first_name and last_name:
                    unique_lobbyists.add(f"{first_name} {last_name}")

        st.metric("Unique Lobbyists", len(unique_lobbyists))


def display_raw_filing(filing: Dict):
    """Display the raw filing data in an expandable section."""
    with st.expander("View Raw Filing Data", expanded=False):
        st.json(filing)


def show_landing_page(filings: List[Dict]):
    """Display the landing page with overview and navigation."""
    st.title("Lobbying Disclosure Analyzer")
    st.markdown(
        "*Analyze lobbying disclosure filings with AI-powered summaries and similarity search*"
    )

    # Overview metrics
    st.markdown("## Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Filings", len(filings))

    with col2:
        filing_years = [f.get("filing_year") for f in filings if f.get("filing_year")]
        if filing_years:
            year_range = f"{min(filing_years)} - {max(filing_years)}"
            st.metric("Year Range", year_range)

    with col3:
        unique_clients = len(
            set(
                f.get("client", {}).get("name", "")
                for f in filings
                if f.get("client", {}).get("name")
            )
        )
        st.metric("Unique Clients", unique_clients)

    with col4:
        total_activities = sum(len(f.get("lobbying_activities", [])) for f in filings)
        st.metric("Total Activities", total_activities)

    st.markdown("---")

    # Navigation cards
    st.markdown("## What would you like to do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Search by Keywords
        Find filings using natural language search with AI-powered similarity matching.
        """)
        if st.button("Start Keyword Search", type="primary", key="landing_search"):
            st.session_state["current_page"] = "search"
            st.rerun()

    with col2:
        st.markdown("""
        ### Browse All Filings
        View all filings in a sortable table format with detailed information.
        """)
        if st.button("Browse Filings Table", type="primary", key="landing_browse"):
            st.session_state["current_page"] = "browse"
            st.rerun()

    with col3:
        st.markdown("""
        ### Analyze Individual Filing
        Get AI-generated summaries and find similar filings for detailed analysis.
        """)
        if st.button("Analyze Filing", type="primary", key="landing_analyze"):
            st.session_state["current_page"] = "analyze"
            st.rerun()


def show_search_page(embedder):
    """Display the search by keyword interface."""
    st.title("Search Filings by Keywords")
    st.markdown(
        "Enter keywords to find filings with similar content using AI-powered semantic search."
    )

    # Search interface
    search_text = st.text_input(
        "Search query:",
        placeholder="e.g., healthcare legislation, defense contracts, environmental policy",
        help="Enter keywords or phrases to search for similar filings",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # Reduced similarity threshold for better results
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.8,
            value=0.3,  # Much lower default
            step=0.05,
            help="Lower values return more diverse results",
        )

    with col2:
        max_results = st.selectbox("Max Results", [5, 10, 15, 20, 25], index=2)

    with col3:
        filter_year = st.selectbox(
            "Filter by Year",
            ["All Years"] + [str(year) for year in range(2020, 2025)],
            index=0,
        )

    if st.button("Search Filings", type="primary") and search_text.strip():
        with st.spinner("Searching for similar filings..."):
            try:
                year_filter = None if filter_year == "All Years" else int(filter_year)

                search_results = embedder.search_filings_by_text(
                    search_text,
                    top_k=max_results,
                    similarity_threshold=similarity_threshold,
                    filter_by_year=year_filter,
                )

                if search_results:
                    st.success(f"Found {len(search_results)} similar filings")
                    display_similar_filings(search_results, [])
                else:
                    st.info(
                        "No similar filings found. Try lowering the similarity threshold or using different keywords."
                    )

            except Exception as e:
                st.error(f"Error performing search: {str(e)}")


def show_browse_page(filings: List[Dict], df: pd.DataFrame):
    """Display the tabular browse interface."""
    st.title("Browse All Filings")
    st.markdown("View and sort all lobbying disclosure filings in a table format.")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        # Year filter
        all_years = sorted(df["Filing Year"].unique())
        selected_years = st.multiselect(
            "Filter by Year",
            options=all_years,
            default=all_years,
            key="browse_year_filter",
        )

    with col2:
        # Client filter
        all_clients = sorted(df["Client"].unique())
        selected_client = st.selectbox(
            "Filter by Client",
            options=["All Clients"] + all_clients,
            key="browse_client_filter",
        )

    with col3:
        # Registrant filter
        all_registrants = sorted(df["Registrant"].unique())
        selected_registrant = st.selectbox(
            "Filter by Registrant",
            options=["All Registrants"] + all_registrants,
            key="browse_registrant_filter",
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_years:
        filtered_df = filtered_df[filtered_df["Filing Year"].isin(selected_years)]

    if selected_client != "All Clients":
        filtered_df = filtered_df[filtered_df["Client"] == selected_client]

    if selected_registrant != "All Registrants":
        filtered_df = filtered_df[filtered_df["Registrant"] == selected_registrant]

    st.markdown(f"Showing {len(filtered_df)} of {len(df)} filings")

    # Sortable columns
    sort_column = st.selectbox(
        "Sort by:",
        options=[
            "Client",
            "Registrant",
            "Filing Year",
            "Filing Period",
            "Amount",
            "Activities",
        ],
        key="browse_sort_column",
    )

    sort_ascending = (
        st.radio(
            "Sort order:", options=["Ascending", "Descending"], key="browse_sort_order"
        )
        == "Ascending"
    )

    # Sort the dataframe
    if sort_column in filtered_df.columns:
        if sort_column == "Amount":
            # Special handling for amount column (remove $ and commas for proper sorting)
            filtered_df["Amount_Numeric"] = (
                filtered_df["Amount"]
                .str.replace("$", "")
                .str.replace(",", "")
                .str.replace("Not reported", "0")
                .astype(float)
            )
            filtered_df = filtered_df.sort_values(
                "Amount_Numeric", ascending=sort_ascending
            )
            filtered_df = filtered_df.drop("Amount_Numeric", axis=1)
        elif sort_column == "Activities":
            filtered_df = filtered_df.sort_values(
                "Activities", ascending=sort_ascending
            )
        else:
            filtered_df = filtered_df.sort_values(sort_column, ascending=sort_ascending)

    # Display the table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Index": st.column_config.NumberColumn("ID", width="small"),
            "Client": st.column_config.TextColumn("Client", width="medium"),
            "Registrant": st.column_config.TextColumn("Registrant", width="medium"),
            "Filing Year": st.column_config.NumberColumn("Year", width="small"),
            "Issue Areas": st.column_config.TextColumn("Issue Areas", width="large"),
            "Amount": st.column_config.TextColumn("Amount", width="small"),
        },
    )

    # Selection for detailed view
    st.markdown("---")
    st.markdown("### View Detailed Filing")

    selected_indices = st.multiselect(
        "Select filing(s) to view in detail (by ID):",
        options=filtered_df["Index"].tolist(),
        max_selections=1,
        key="browse_detailed_selection",
    )

    if selected_indices:
        if st.button("View Selected Filing", type="primary"):
            st.session_state["selected_filing_idx"] = selected_indices[0]
            st.session_state["current_page"] = "analyze"
            st.rerun()


def show_analyze_page(filings: List[Dict], embedder):
    """Display the individual filing analysis interface."""
    st.title("Analyze Individual Filing")

    # Filing selection
    st.markdown("### Select Filing")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Check if we have a pre-selected filing
        if "selected_filing_idx" in st.session_state:
            default_idx = st.session_state["selected_filing_idx"]
        else:
            default_idx = 0

        filing_options = {}
        for i, filing in enumerate(filings):
            shorthand = create_filing_shorthand(filing)
            original_shorthand = shorthand
            counter = 1
            while shorthand in filing_options:
                shorthand = f"{original_shorthand} ({counter})"
                counter += 1
            filing_options[shorthand] = i

        selected_shorthand = st.selectbox(
            "Choose filing:",
            options=list(filing_options.keys()),
            index=list(filing_options.values()).index(default_idx)
            if default_idx in filing_options.values()
            else 0,
            help="Filings are shown as: Client | Registrant | Year Period",
        )

        selected_index = filing_options[selected_shorthand]
        selected_filing = filings[selected_index]

    with col2:
        if st.button("Random Filing", type="secondary"):
            random_idx = random.randint(0, len(filings) - 1)
            st.session_state["selected_filing_idx"] = random_idx
            st.rerun()

    # Clear the selected filing from session state after use
    if "selected_filing_idx" in st.session_state:
        del st.session_state["selected_filing_idx"]

    # Display filing analysis
    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate and display summary
        with st.spinner("Generating AI summary..."):
            try:
                summarizer = get_summarizer()
                summary = summarizer.summarize_filing(selected_filing)
                display_summary(summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                st.info("Showing filing details without AI summary...")

        # Find Similar Filings section
        st.markdown("---")

        if embedder:
            st.markdown("### Find Similar Filings")

            col_btn, col_params = st.columns([1, 2])

            with col_btn:
                find_similar = st.button("Find Similar Filings", type="primary")

            with col_params:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.4,  # Lower default threshold
                    step=0.05,
                    help="Lower values return more diverse results",
                )
                max_results = st.selectbox(
                    "Max Results",
                    [5, 10, 15, 20],
                    index=1,
                )

            if find_similar:
                with st.spinner("Finding similar filings..."):
                    try:
                        similar_filings = embedder.find_similar_filings(
                            selected_filing,
                            target_filing_idx=selected_index,
                            top_k=max_results,
                            similarity_threshold=similarity_threshold,
                        )

                        display_similar_filings(similar_filings, filings)

                    except Exception as e:
                        st.error(f"Error finding similar filings: {str(e)}")
        else:
            st.markdown("### Find Similar Filings")
            st.warning(
                "Similarity search is not available. Embeddings need to be generated first."
            )

        # Raw data
        st.markdown("---")
        display_raw_filing(selected_filing)

    with col2:
        # Filing metadata
        display_filing_metadata(selected_filing)

        # Additional information
        st.markdown("### Additional Information")

        filing_url = selected_filing.get("filing_document_url")
        if filing_url:
            st.markdown(f"[View Original Filing]({filing_url})")

        posted_by = selected_filing.get("posted_by_name")
        posted_date = selected_filing.get("dt_posted")
        if posted_by:
            st.write(f"**Posted by:** {posted_by}")
        if posted_date:
            st.write(f"**Posted on:** {posted_date[:10]}")


def main():
    """Main Streamlit app."""

    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "landing"

    # Load data
    with st.spinner("Loading lobbying disclosure data..."):
        filings = load_filings_data()

    if not filings:
        st.error("No filing data available. Please check your data file.")
        return

    # Initialize embedder
    embedder = get_embedder()

    # Create dataframe for table view
    df = create_filings_dataframe(filings)

    # Navigation
    st.sidebar.title("Navigation")

    page_options = {
        "Home": "landing",
        "Search by Keywords": "search",
        "Browse All Filings": "browse",
        "Analyze Individual Filing": "analyze",
    }

    # Show current page in sidebar
    current_page_name = None
    for name, page_id in page_options.items():
        if page_id == st.session_state["current_page"]:
            current_page_name = name
            break

    if current_page_name:
        st.sidebar.markdown(f"**Current:** {current_page_name}")

    st.sidebar.markdown("---")

    # Navigation buttons
    for name, page_id in page_options.items():
        if st.sidebar.button(name, key=f"nav_{page_id}"):
            st.session_state["current_page"] = page_id
            st.rerun()

    # Dataset info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.metric("Total Filings", len(filings))

    if embedder:
        st.sidebar.success("Similarity search ready")
        stats = embedder.get_embedding_stats()
        st.sidebar.info(f"Embeddings: {stats.get('total_filings', 0)} filings")
    else:
        st.sidebar.info("Similarity search unavailable")

    # Display current page
    if st.session_state["current_page"] == "landing":
        show_landing_page(filings)
    elif st.session_state["current_page"] == "search":
        if embedder:
            show_search_page(embedder)
        else:
            st.error(
                "Search functionality requires embeddings. Please generate embeddings first."
            )
    elif st.session_state["current_page"] == "browse":
        show_browse_page(filings, df)
    elif st.session_state["current_page"] == "analyze":
        show_analyze_page(filings, embedder)


if __name__ == "__main__":
    main()
