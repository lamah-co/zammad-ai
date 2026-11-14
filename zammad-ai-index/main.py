"""Entry point for the Zammad AI index job."""

# ruff: noqa: E402 (no import at top level) suppressed on this file as we need to inject the truststore before importing the other modules

from dotenv import load_dotenv
from truststore import inject_into_ssl

inject_into_ssl()
load_dotenv()

from logging import Logger
from uuid import UUID

from qdrant_client.models import Record

from job.data.processing import filter_for_changed_data, prepare_qdrant_data
from job.data.retrieval import get_answers_data, retrieve_answer_ids, retrieve_deleted_answer_ids
from job.docs.fetch import checkout
from job.docs.mapping import build_plan
from job.docs.sync import archive_category, sync_docs
from job.law.ingest import ingest_law
from job.models.qdrant import QdrantDocumentItem
from job.models.zammad import KnowledgeBaseAnswer
from job.projects import ProjectConfig, load_projects
from job.qdrant.qdrant import QdrantKBClient, ensure_collection
from job.settings.settings import ZammadAIIndexSettings, get_settings
from job.settings.zammad import ZammadAPISettings, ZammadEAISettings
from job.utils.logging import getLogger
from job.zammad.api import ZammadAPIClient
from job.zammad.eai import ZammadEAIClient
from job.zammad.kb_writer import KBWriter

settings: ZammadAIIndexSettings = get_settings()
logger: Logger = getLogger("zammad-ai-index")


def run_indexing(
    qdrant_client: QdrantKBClient,
    zammad_client: ZammadAPIClient | ZammadEAIClient,
    snapshot_already_created: bool = False,
    category_ids: list[int] | None = None,
    project: str | None = None,
) -> None:
    """Execute the complete indexing workflow.

    Performs a nine-step process:
    0. Retrieves answer IDs of filtered categories if category filtering is enabled
    1. Retrieves answer IDs for processing based on configured criteria
    2. Fetches detailed answer data for the retrieved answer IDs
    3. Gets all points from Qdrant to compare with new data
    4. Prepares and filters data for Qdrant, keeping only new or changed documents
    5. Checks for deleted documents in Zammad and retrieves their IDs
    6. Creates a snapshot in Qdrant before indexing to allow rollback if needed
    7. Adds new and changed documents to Qdrant in batches for optimal performance
    8. Deletes documents from Qdrant that correspond to deleted answer IDs in Zammad

    The method handles errors gracefully and ensures cleanup of resources.
    Process completion status is logged at info level.

    Raises:
        Exception: Critical errors during indexing are logged and re-raised.

    """
    logger.info("Starting indexing process...")

    try:
        # Step 0: Retrieve answer IDs of filtered categories if category filtering is enabled
        answer_ids_in_category: list[int] = zammad_client.get_answer_ids_of_categories(
            settings.zammad.category_ids if category_ids is None else category_ids
        )

        # Step 1: Retrieve answer IDs for processing
        answer_ids: list[int] = retrieve_answer_ids(zammad_client, answer_ids_in_category)
        answers: dict[int, KnowledgeBaseAnswer] = {}
        if not answer_ids:
            logger.info("No answer IDs found for processing.")
        else:
            logger.info("Retrieved %d answer IDs for processing.", len(answer_ids))

            # Step 2: Fetch detailed answer data
            answers: dict[int, KnowledgeBaseAnswer] = get_answers_data(answer_ids, zammad_client)

            if not answers:
                logger.info("No answer data retrieved for the provided answer IDs.")
            else:
                logger.info("Fetched data for %d answers.", len(answers))

        # Step 3: Get all points from Qdrant
        all_points: list[Record] = qdrant_client.get_all_points()

        # Step 4: Prepare and filter data for Qdrant
        qdrant_items: list[QdrantDocumentItem] = _prepare_and_filter_data(answers, all_points, zammad_client, project)

        # Step 5: Check for deleted documents
        deleted_answer_ids: list[UUID] = retrieve_deleted_answer_ids(all_points, zammad_client, answer_ids_in_category)

        if not qdrant_items:
            logger.info("No new or changed documents to index.")
            if deleted_answer_ids:
                logger.info(
                    "However, %d deleted answer IDs detected that will be removed from Qdrant.", len(deleted_answer_ids)
                )
            else:
                logger.info("No deleted answer IDs detected. Exiting.")
                return
        else:
            logger.info("Prepared %d documents for Qdrant indexing.", len(qdrant_items))

        # Step 6: Create Qdrant snapshot if not already created by caller
        snapshot_success: bool = True
        if not snapshot_already_created:
            snapshot_success = qdrant_client.create_snapshot()

        if snapshot_success:
            logger.info("Successfully created or skipped Qdrant snapshot before indexing.")
        else:
            logger.warning("Failed to create Qdrant snapshot before indexing. Exiting.")
            return

        # Step 7: Add documents to Qdrant
        if qdrant_items:
            success: bool = add_documents_to_qdrant(qdrant_items, qdrant_client)

            if success:
                logger.info("Successfully indexed documents into Qdrant.")
            else:
                logger.error("Failed to index documents into Qdrant.")
                return

        # Step 8: Delete documents from Qdrant for deleted answer IDs
        if deleted_answer_ids:
            try:
                qdrant_client.delete_points_by_ids(deleted_answer_ids)
                logger.info(
                    "Successfully deleted %d documents from Qdrant for deleted answer IDs.", len(deleted_answer_ids)
                )
            except Exception:
                logger.error("Failed to delete points for deleted answer IDs: %s", deleted_answer_ids, exc_info=True)

        logger.info("Indexing process completed successfully.")

    except Exception:
        logger.error("Critical error during indexing process", exc_info=True)
        raise


def _prepare_and_filter_data(
    answers: dict[int, KnowledgeBaseAnswer],
    all_points: list[Record],
    zammad_client: ZammadAPIClient | ZammadEAIClient,
    project: str | None = None,
) -> list[QdrantDocumentItem]:
    """Prepare data for Qdrant and filter for changed documents only.

    Transforms knowledge base answers into Qdrant document format with metadata,
    then filters to include only documents that have changed since last indexing
    to avoid unnecessary re-processing.

    Args:
        answers: Dictionary mapping answer IDs to KnowledgeBaseAnswer objects
        all_points: List of all points in the Qdrant collection
        zammad_client: Zammad API client instance
        project: Project key stored with every vector, or None outside multi-project mode

    Returns:
        List of QdrantDocumentItem objects ready for indexing, containing only
        new or changed documents based on content hash comparison.

    """
    # Prepare data for Qdrant indexing
    qdrant_data: list[QdrantDocumentItem] = prepare_qdrant_data(
        answers=answers,
        client=zammad_client,
        project=project,
    )
    logger.info("Prepared data for %d answers for Qdrant indexing.", len(qdrant_data))

    # Filter to only include changed documents
    filtered_items: list[QdrantDocumentItem] = filter_for_changed_data(
        new_qdrant_data=qdrant_data,
        all_points=all_points,
    )

    return filtered_items


def add_documents_to_qdrant(qdrant_items: list[QdrantDocumentItem], qdrant_client: QdrantKBClient) -> bool:
    """Add documents to Qdrant in batches for optimal performance.

    Args:
        qdrant_items: List of QdrantDocumentItem objects containing id, content, and metadata for indexing
        qdrant_client: Instance of QdrantKBClient for interacting with Qdrant


    Returns:
        True if all documents were successfully added, False otherwise

    """
    if not qdrant_items:
        logger.info("No documents to add to Qdrant.")
        return True

    batch_size: int = settings.index.batch_size

    total_batches: int = (len(qdrant_items) + batch_size - 1) // batch_size
    successful_batches: int = 0

    logger.info("Starting to add %d documents to Qdrant in %d batches.", len(qdrant_items), total_batches)

    for i in range(0, len(qdrant_items), batch_size):
        batch_number: int = i // batch_size + 1
        batch: list[QdrantDocumentItem] = qdrant_items[i : i + batch_size]

        try:
            logger.info("Processing batch %d/%d with %d documents", batch_number, total_batches, len(batch))

            qdrant_client.add_documents(batch)

            logger.debug("Successfully added batch %d/%d to Qdrant", batch_number, total_batches)
            successful_batches += 1
        except Exception:
            logger.error("Exception occurred while processing batch %d/%d", batch_number, total_batches, exc_info=True)

    logger.info("Successfully processed %d/%d batches.", successful_batches, total_batches)
    return successful_batches == total_batches


def cleanup(zammad_client, qdrant_client) -> None:
    """Clean up resources and close connections.

    Properly closes Zammad client and Qdrant client connections to prevent
    resource leaks. This method is called in the finally block of run_indexing
    to ensure cleanup occurs even if errors are encountered.
    """
    cleanup_errors: list[Exception] = []

    if zammad_client:
        try:
            zammad_client.close()
            logger.debug("Closed Zammad client connection")
        except Exception as e:
            logger.error("Failed to close Zammad client connection", exc_info=True)
            cleanup_errors.append(e)

    if qdrant_client:
        try:
            qdrant_client.close()
            logger.debug("Closed Qdrant client connection")
        except Exception as e:
            logger.error("Failed to close Qdrant client connection", exc_info=True)
            cleanup_errors.append(e)

    if cleanup_errors:
        logger.warning("Cleanup finished with %d error(s).", len(cleanup_errors))


def _index_single() -> None:
    """Index the whole configured KB (or its category_ids) into the configured collection, as before projects."""
    zammad_client: ZammadAPIClient | ZammadEAIClient | None = None
    qdrant_client: QdrantKBClient | None = None
    try:
        # Initialize Zammad client based on configuration
        if isinstance(settings.zammad, ZammadAPISettings):
            zammad_client = ZammadAPIClient(settings.zammad)
            logger.info("Initialized Zammad API-Client")
        elif isinstance(settings.zammad, ZammadEAISettings):
            zammad_client = ZammadEAIClient(settings.zammad)
            logger.info("Initialized Zammad EAI-Client")
        else:
            raise ValueError(f"Unsupported Zammad client type: {settings.zammad.type}")

        qdrant_client = QdrantKBClient(settings)
        logger.info("Initialized Qdrant client")

        # Create a single Qdrant snapshot before any writes (laws + KB), so a failure can be rolled back.
        snapshot_created = False
        if settings.laws:
            logger.info("Creating Qdrant snapshot before law ingestion")
            snapshot_created = qdrant_client.create_snapshot()
            if not snapshot_created:
                logger.warning("Failed to create Qdrant snapshot before law ingestion. Exiting.")
                return

            logger.info("Starting law ingestion for %d configured law(s)", len(settings.laws))
            for law in settings.laws:
                ingest_law(law, qdrant_client)
            logger.info("Completed law ingestion stage")

        run_indexing(qdrant_client, zammad_client, snapshot_already_created=snapshot_created)
    finally:
        cleanup(zammad_client, qdrant_client)


def _api_settings() -> ZammadAPISettings:
    if not isinstance(settings.zammad, ZammadAPISettings):
        raise ValueError("Projects and the docs sync need zammad.type: api")
    return settings.zammad


def _load_projects() -> dict[str, ProjectConfig]:
    if settings.projects_dir is None:
        raise ValueError("projects_dir (ZAMMAD_AI_PROJECTS_DIR) is required for projects")
    projects = load_projects(settings.projects_dir)
    if settings.project is not None and settings.project not in projects:
        raise ValueError(f"Unknown project {settings.project!r}; known: {', '.join(projects) or 'none'}")
    return projects


def _root_category_id(project: ProjectConfig) -> int:
    reader = KBWriter(_api_settings(), _api_settings().auth_token)
    try:
        tree = reader.read_tree()
        return next(
            c.id for c in tree.categories.values() if c.parent_id is None and c.title == project.kb.root_category
        )
    except StopIteration:
        raise ValueError(
            f"KB category {project.kb.root_category!r} of project {project.key} does not exist; run sync-docs first"
        ) from None
    finally:
        reader.client.close()


def index_project(project: ProjectConfig) -> None:
    """Index one project's KB category tree into its own Qdrant collection."""
    root_id = _root_category_id(project)
    project_settings = settings.model_copy(
        update={"qdrant": settings.qdrant.model_copy(update={"collection_name": project.qdrant.collection_name})}
    )
    if ensure_collection(project_settings.qdrant):
        logger.info(f"Created Qdrant collection {project.qdrant.collection_name} for project {project.key}")
    zammad_client = ZammadAPIClient(_api_settings())
    qdrant_client: QdrantKBClient | None = None
    try:
        qdrant_client = QdrantKBClient(project_settings)
        qdrant_client.ensure_project_index()
        logger.info(f"Indexing project {project.key} (category {root_id}) into {project.qdrant.collection_name}")
        run_indexing(qdrant_client, zammad_client, category_ids=[root_id], project=project.key)
    finally:
        cleanup(zammad_client, qdrant_client)


def sync_project_docs(project: ProjectConfig) -> None:
    """Sync a project's docs repo into its KB category tree, then index it when asked and something changed."""
    if project.docs is None:
        raise ValueError(f"Project {project.key} has no docs repository to sync")
    if settings.docs_github_token is None or settings.kb_editor_token is None:
        raise ValueError("sync-docs needs ZAMMAD_AI_DOCS_GITHUB_TOKEN and ZAMMAD_AI_KB_EDITOR_TOKEN")
    writer = KBWriter(_api_settings(), settings.kb_editor_token)
    try:
        with checkout(project.docs.repo, project.docs.branch, settings.docs_github_token) as (root, sha):
            logger.info(f"Syncing {project.docs.repo}@{sha} into KB category {project.kb.root_category!r}")
            plan = build_plan(project, root)
        result = sync_docs(
            plan,
            root_title=project.kb.root_category,
            writer=writer,
            state_file=settings.sync_state_dir / f"{project.key}.json",
            dry_run=settings.dry_run,
        )
    finally:
        writer.client.close()
    prefix = "Would " if settings.dry_run else ""
    for action, subject in result.actions:
        if action != "unchanged":
            logger.info(f"{prefix}{action}: {subject}")
    summary = ", ".join(f"{count} {action}" for action, count in sorted(result.counts.items())) or "nothing"
    logger.info(f"Docs sync for {project.key} at {sha}{' (dry run)' if settings.dry_run else ''}: {summary}")
    if settings.then_index and not settings.dry_run:
        if result.changed:
            index_project(project)
        else:
            logger.info(f"KB of project {project.key} unchanged; skipping indexing")


def archive_manual_category() -> None:
    """Archive every answer in archive_category_id, to retire hand-written answers the docs now provide."""
    if settings.archive_category_id is None or settings.kb_editor_token is None:
        raise ValueError("archive-category needs archive_category_id and ZAMMAD_AI_KB_EDITOR_TOKEN")
    writer = KBWriter(_api_settings(), settings.kb_editor_token)
    try:
        result = archive_category(settings.archive_category_id, writer=writer, dry_run=settings.dry_run)
    finally:
        writer.client.close()
    for _, title in result.actions:
        logger.info(f"{'Would archive' if settings.dry_run else 'Archived'}: {title}")
    logger.info(f"Archived {len(result.actions)} answer(s) in category {settings.archive_category_id}")


def main() -> None:
    """Run one command: index (default), sync-docs or archive-category."""
    try:
        if settings.command == "sync-docs":
            projects = _load_projects()
            if settings.project is None:
                raise ValueError("sync-docs needs --project")
            sync_project_docs(projects[settings.project])
        elif settings.command == "archive-category":
            archive_manual_category()
        elif settings.projects_dir is None:
            _index_single()
        else:
            projects = _load_projects()
            for key in [settings.project] if settings.project else list(projects):
                index_project(projects[key])
    except Exception:
        getLogger("zammad-ai-index").error("Application encountered a critical error", exc_info=True)
        raise


if __name__ == "__main__":
    main()
