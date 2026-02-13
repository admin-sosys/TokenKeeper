# Tests for PLT-01: ChromaDB works on Windows 11
#
# These tests validate that ChromaDB PersistentClient operations work
# correctly on Windows 11. They serve as regression checks that can be
# re-run in later phases to ensure ChromaDB remains functional.
#
# Key finding: RustBindingsAPI (default) works on this system without
# hanging. The SegmentAPI bypass is not needed (chromadb==1.5.0).

import pytest

import chromadb


@pytest.fixture
def chromadb_client(tmp_path):
    """Create a PersistentClient in a temp directory.

    Uses the default RustBindingsAPI which works on this Windows 11 system.
    The tmp_path fixture provides a unique temp directory per test that
    pytest automatically cleans up after the test completes.
    """
    persist_dir = str(tmp_path / "chroma_data")
    client = chromadb.PersistentClient(path=persist_dir)
    yield client
    # Cleanup: delete all collections to release resources
    for col in client.list_collections():
        try:
            client.delete_collection(col)
        except Exception:
            pass


@pytest.fixture
def test_collection(chromadb_client):
    """Create a test collection with sample data.

    Provides a pre-populated collection for tests that need existing data.
    Deletes the collection after the test completes.
    """
    collection = chromadb_client.get_or_create_collection("test_collection")
    collection.add(
        ids=["doc1", "doc2", "doc3"],
        documents=[
            "ChromaDB platform validation document one",
            "Windows 11 compatibility testing document two",
            "Vector database verification document three",
        ],
        metadatas=[
            {"source": "test", "category": "validation"},
            {"source": "test", "category": "compatibility"},
            {"source": "validation", "category": "verification"},
        ],
    )
    yield collection
    try:
        chromadb_client.delete_collection("test_collection")
    except Exception:
        pass


class TestClientCreation:
    """Tests for ChromaDB client initialization and basic operations."""

    def test_client_creates_with_persistent_client(self, chromadb_client):
        """Verify PersistentClient creates successfully on Windows 11.

        This is the most fundamental test: if ChromaDB cannot initialize
        its client, nothing else will work. The chroma#5937 issue caused
        the client to hang during creation on some Windows systems.
        """
        assert chromadb_client is not None

    def test_heartbeat_returns(self, chromadb_client):
        """Verify heartbeat returns a non-zero value.

        Heartbeat confirms the ChromaDB internal server is running and
        responsive. A hanging heartbeat would indicate the Rust FFI
        deadlock from chroma#5937.
        """
        heartbeat = chromadb_client.heartbeat()
        assert heartbeat is not None
        assert heartbeat > 0

    def test_get_version_is_non_empty(self, chromadb_client):
        """Verify get_version returns a non-empty version string.

        Confirms the client can communicate with the internal server
        and retrieve metadata about the running ChromaDB instance.
        """
        version = chromadb_client.get_version()
        assert version is not None
        assert len(version) > 0
        assert version == "1.5.0"


class TestCollectionCRUD:
    """Tests for collection-level create, read, update, delete operations."""

    def test_collection_crud(self, chromadb_client):
        """Verify full collection lifecycle: create, populate, query, delete.

        This test exercises the complete lifecycle of a ChromaDB collection
        on Windows 11. Each operation must complete without hanging, which
        validates that the Rust FFI bindings work correctly.
        """
        # Create
        collection = chromadb_client.get_or_create_collection("crud_test")
        assert collection is not None
        assert collection.name == "crud_test"

        # Add documents
        collection.add(
            ids=["c1", "c2", "c3"],
            documents=[
                "First test document for CRUD",
                "Second test document for CRUD",
                "Third test document for CRUD",
            ],
            metadatas=[
                {"index": 1},
                {"index": 2},
                {"index": 3},
            ],
        )
        assert collection.count() == 3

        # Get by ID
        result = collection.get(ids=["c1"], include=["documents", "metadatas"])
        assert len(result["ids"]) == 1
        assert result["documents"][0] == "First test document for CRUD"
        assert result["metadatas"][0]["index"] == 1

        # Delete one document
        collection.delete(ids=["c3"])
        assert collection.count() == 2

        # Delete collection
        chromadb_client.delete_collection("crud_test")
        collections = chromadb_client.list_collections()
        assert len(collections) == 0


class TestQueryOperations:
    """Tests for ChromaDB query/search functionality."""

    def test_query_returns_results(self, test_collection):
        """Verify semantic query returns expected result structure.

        Query is the core operation for our RAG system. The query must
        return documents ranked by similarity, with ids, documents, and
        distances all populated. This validates that the embedding model
        and HNSW index are working correctly through the Rust bindings.
        """
        result = test_collection.query(
            query_texts=["platform validation testing"],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )

        # Verify structure
        assert "ids" in result
        assert "documents" in result
        assert "distances" in result
        assert "metadatas" in result

        # Verify content
        assert len(result["ids"][0]) == 2
        assert len(result["documents"][0]) == 2
        assert len(result["distances"][0]) == 2
        assert len(result["metadatas"][0]) == 2

        # Distances should be non-negative
        for distance in result["distances"][0]:
            assert distance >= 0

    def test_collection_with_metadata_filter(self, test_collection):
        """Verify metadata filtering works in queries.

        Metadata filtering is essential for scoping searches to specific
        document types, sources, or categories. This validates that
        ChromaDB's where clause filtering works with the Rust bindings
        on Windows 11.
        """
        # Filter by source="test" should return 2 documents
        result = test_collection.get(
            where={"source": "test"},
            include=["documents", "metadatas"],
        )
        assert len(result["ids"]) == 2
        for metadata in result["metadatas"]:
            assert metadata["source"] == "test"

        # Filter by source="validation" should return 1 document
        result = test_collection.get(
            where={"source": "validation"},
            include=["documents", "metadatas"],
        )
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["source"] == "validation"


class TestPersistence:
    """Tests for data persistence across client restarts."""

    def test_persistence_across_restart(self, tmp_path):
        """Verify data survives client restart on Windows 11.

        This is the critical persistence test. Our RAG system indexes
        documents once and queries them many times. If ChromaDB cannot
        persist data to disk and reload it after a client restart, the
        entire indexing pipeline would need to re-run on every startup.

        Tests: create client -> add data -> delete client -> new client
        -> verify data is still present.
        """
        persist_dir = str(tmp_path / "persist_test")

        # Phase 1: Create client and add data
        client1 = chromadb.PersistentClient(path=persist_dir)
        collection = client1.get_or_create_collection("persist_collection")
        collection.add(
            ids=["p1", "p2"],
            documents=["Persistent document one", "Persistent document two"],
            metadatas=[{"type": "persist"}, {"type": "persist"}],
        )
        assert collection.count() == 2

        # Phase 2: Delete client reference (simulates process exit)
        del client1

        # Phase 3: Create NEW client pointing to same directory
        client2 = chromadb.PersistentClient(path=persist_dir)
        restored_collection = client2.get_collection("persist_collection")

        # Phase 4: Verify data persisted
        assert restored_collection.count() == 2

        result = restored_collection.get(
            ids=["p1", "p2"],
            include=["documents", "metadatas"],
        )
        assert len(result["ids"]) == 2
        assert result["documents"][0] == "Persistent document one"
        assert result["documents"][1] == "Persistent document two"
        assert result["metadatas"][0]["type"] == "persist"
        assert result["metadatas"][1]["type"] == "persist"

        # Cleanup
        client2.delete_collection("persist_collection")


class TestUpdateOperations:
    """Tests for document update operations."""

    def test_document_update(self, test_collection):
        """Verify document content can be updated in-place.

        Updates are needed when source documents change. The RAG system
        must be able to update existing embeddings without deleting and
        re-adding the entire document.
        """
        # Update doc1 content
        test_collection.update(
            ids=["doc1"],
            documents=["Updated content for document one"],
        )

        # Verify update took effect
        result = test_collection.get(ids=["doc1"], include=["documents"])
        assert result["documents"][0] == "Updated content for document one"

        # Verify other documents unchanged
        result = test_collection.get(ids=["doc2"], include=["documents"])
        assert (
            result["documents"][0]
            == "Windows 11 compatibility testing document two"
        )
