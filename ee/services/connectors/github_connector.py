import logging
import os
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs

from github import Github, GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile
import requests

from ee.config import EESettings, get_ee_settings
from .base_connector import BaseConnector, ConnectorAuthStatus, ConnectorFile

logger = logging.getLogger(__name__)

# Initialize mimetypes
mimetypes.init()

class GitHubConnector(BaseConnector):
    connector_type = "github"

    def __init__(self, user_morphik_id: str):
        super().__init__(user_morphik_id)
        self.ee_settings: EESettings = get_ee_settings()
        self.client: Optional[Github] = None
        self._load_credentials()

    def _get_user_token_path(self) -> Path:
        token_dir = Path(self.ee_settings.GITHUB_TOKEN_STORAGE_PATH)
        os.makedirs(token_dir, exist_ok=True)
        return token_dir / f"github_token_{self.user_morphik_id}.txt"

    def _save_credentials(self, access_token: str) -> None:
        token_path = self._get_user_token_path()
        try:
            with open(token_path, "w") as token_file:
                token_file.write(access_token)
            logger.info(f"Successfully saved GitHub token for user {self.user_morphik_id}")
        except Exception as e:
            logger.error(f"Failed to save GitHub token for user {self.user_morphik_id}: {e}")

    def _load_credentials(self) -> None:
        token_path = self._get_user_token_path()
        if token_path.exists():
            try:
                with open(token_path, "r") as token_file:
                    access_token = token_file.read().strip()
                self.client = Github(access_token)
                # Test the connection
                self.client.get_user().login
                logger.info(f"Successfully loaded GitHub token for user {self.user_morphik_id}")
            except Exception as e:
                logger.error(f"Failed to load GitHub token for user {self.user_morphik_id}: {e}")
                self.client = None
        else:
            logger.info(f"No GitHub token found for user {self.user_morphik_id}")
            self.client = None

    async def get_auth_status(self) -> ConnectorAuthStatus:
        if not self.ee_settings.GITHUB_CLIENT_ID or not self.ee_settings.GITHUB_CLIENT_SECRET:
            return ConnectorAuthStatus(
                is_authenticated=False,
                message="GitHub OAuth credentials not configured"
            )

        if self.client:
            try:
                # Test the connection
                self.client.get_user().login
                return ConnectorAuthStatus(
                    is_authenticated=True,
                    message="Successfully connected to GitHub"
                )
            except Exception as e:
                logger.error(f"GitHub connection test failed for user {self.user_morphik_id}: {e}")

        return ConnectorAuthStatus(
            is_authenticated=False,
            message="Not authenticated with GitHub"
        )

    async def initiate_auth(self) -> Dict[str, Any]:
        """Initiate OAuth flow for GitHub."""
        if not self.ee_settings.GITHUB_CLIENT_ID or not self.ee_settings.GITHUB_CLIENT_SECRET:
            raise ValueError("GitHub OAuth credentials not configured")

        params = {
            "client_id": self.ee_settings.GITHUB_CLIENT_ID,
            "redirect_uri": self.ee_settings.GITHUB_REDIRECT_URI,
            "scope": " ".join(self.ee_settings.GITHUB_SCOPES),
            "state": self.user_morphik_id,  # Use user ID as state
        }

        auth_url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"

        return {
            "authorization_url": auth_url,
            "state": self.user_morphik_id
        }

    async def finalize_auth(self, auth_response_data: Dict[str, Any]) -> bool:
        """Handle OAuth callback and token exchange."""
        # Get code from authorization_response_url query parameters
        authorization_response_url = auth_response_data.get("authorization_response_url")
        if not authorization_response_url:
            raise ValueError("No authorization response URL provided")

        # Parse the code from the URL query parameters
        parsed_url = urlparse(authorization_response_url)
        params = parse_qs(parsed_url.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]

        if not code:
            raise ValueError("No authorization code provided")

        if state != self.user_morphik_id:
            raise ValueError("Invalid state parameter")

        # Exchange code for access token
        token_url = "https://github.com/login/oauth/access_token"
        data = {
            "client_id": self.ee_settings.GITHUB_CLIENT_ID,
            "client_secret": self.ee_settings.GITHUB_CLIENT_SECRET,
            "code": code,
            "redirect_uri": self.ee_settings.GITHUB_REDIRECT_URI,
        }
        headers = {"Accept": "application/json"}

        try:
            response = requests.post(token_url, data=data, headers=headers)
            response.raise_for_status()
            token_data = response.json()
            
            if "error" in token_data:
                raise ValueError(f"GitHub OAuth error: {token_data['error']}")

            access_token = token_data.get("access_token")
            if not access_token:
                raise ValueError("No access token in response")

            # Test the token
            test_client = Github(access_token)
            test_client.get_user().login

            # If successful, save the token
            self._save_credentials(access_token)
            self._load_credentials()  # Reload to initialize the client
            return True

        except Exception as e:
            logger.error(f"Failed to exchange GitHub code for token: {e}")
            raise ValueError(f"Failed to complete GitHub OAuth: {str(e)}")

    def _guess_mime_type(self, filename: str) -> str:
        """Guess the MIME type from the file extension."""
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            # Default to binary if we can't detect the type
            mime_type = 'application/octet-stream'
        return mime_type

    def _escape_xml_content(self, content: str) -> str:
        """Escape XML special characters and wrap in CDATA if needed."""
        # First try basic XML escaping
        escaped = (content.replace('&', '&amp;')
                         .replace('<', '&lt;')
                         .replace('>', '&gt;')
                         .replace('"', '&quot;')
                         .replace("'", '&apos;'))
        
        # If content contains problematic characters or is very long, use CDATA
        if (len(escaped) > 50000 or 
            any(ord(c) < 32 and c not in '\t\n\r' for c in content[:1000])):
            # Use CDATA section for complex content
            # Escape any existing ]]> sequences in the content
            cdata_content = content.replace(']]>', ']]&gt;')
            return f"<![CDATA[{cdata_content}]]>"
        
        return escaped

    async def list_files(
        self, path: Optional[str] = None, page_token: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """List repositories and files. Path format: 'owner/repo/path/to/folder'"""
        if not self.client:
            raise ConnectionError("Not authenticated with GitHub")

        try:
            if not path:
                # List repositories
                repos = []
                for repo in self.client.get_user().get_repos():
                    repos.append(ConnectorFile(
                        id=f"{repo.owner.login}/{repo.name}",
                        name=repo.name,
                        is_folder=True,
                        mime_type="application/x-directory",  # Standard MIME type for directories
                        modified_date=repo.updated_at.isoformat() if repo.updated_at else None
                    ))
                return {"files": repos, "next_page_token": None}
            else:
                # Parse path components
                parts = path.split('/')
                if len(parts) < 2:
                    raise ValueError("Invalid path format. Expected: owner/repo/[path/to/folder]")
                
                owner, repo_name, *file_path = parts
                repo = self.client.get_repo(f"{owner}/{repo_name}")
                
                # Get contents of the specified path
                path_in_repo = '/'.join(file_path) if file_path else ''
                contents = repo.get_contents(path_in_repo)
                
                files = []
                if not isinstance(contents, list):
                    contents = [contents]
                
                for item in contents:
                    mime_type = "application/x-directory" if item.type == "dir" else self._guess_mime_type(item.name)
                    files.append(ConnectorFile(
                        id=f"{owner}/{repo_name}/{item.path}",
                        name=item.name,
                        is_folder=item.type == "dir",
                        size=item.size if item.type == "file" else None,
                        mime_type=mime_type,
                        modified_date=None  # Would need additional API call to get this
                    ))
                
                return {"files": files, "next_page_token": None}

        except Exception as e:
            logger.error(f"Error listing GitHub files: {e}")
            raise ConnectionError(f"Failed to list GitHub files: {str(e)}")

    async def download_file_by_id(self, file_id: str) -> Optional[BytesIO]:
        """Download a file from GitHub. file_id format: 'owner/repo/path/to/file'"""
        if not self.client:
            return None

        try:
            # Parse file ID components
            parts = file_id.split('/')
            if len(parts) < 3:
                logger.error(f"Invalid file ID format: {file_id}")
                return None
            
            owner, repo_name, *file_path = parts
            repo = self.client.get_repo(f"{owner}/{repo_name}")
            
            # Get the file content
            file_content = repo.get_contents('/'.join(file_path))
            if isinstance(file_content, list):
                logger.error(f"File ID points to a directory: {file_id}")
                return None
            
            # Download and return the content
            content = file_content.decoded_content
            return BytesIO(content)

        except Exception as e:
            logger.error(f"Error downloading GitHub file {file_id}: {e}")
            return None

    async def get_file_metadata_by_id(self, file_id: str) -> Optional[ConnectorFile]:
        """Get metadata for a GitHub file. file_id format: 'owner/repo/path/to/file'"""
        if not self.client:
            return None

        try:
            # Parse file ID components
            parts = file_id.split('/')
            if len(parts) < 3:
                logger.error(f"Invalid file ID format: {file_id}")
                return None
            
            owner, repo_name, *file_path = parts
            repo = self.client.get_repo(f"{owner}/{repo_name}")
            
            # Get the file content
            file_content = repo.get_contents('/'.join(file_path))
            if isinstance(file_content, list):
                logger.error(f"File ID points to a directory: {file_id}")
                return None
            
            mime_type = self._guess_mime_type(file_content.name)
            
            return ConnectorFile(
                id=file_id,
                name=file_content.name,
                is_folder=False,
                size=file_content.size,
                mime_type=mime_type,
                modified_date=None  # Would need additional API call to get this
            )

        except Exception as e:
            logger.error(f"Error getting GitHub file metadata for {file_id}: {e}")
            return None

    async def disconnect(self) -> bool:
        """Remove stored GitHub credentials."""
        token_path = self._get_user_token_path()
        if token_path.exists():
            try:
                token_path.unlink()
                logger.info(f"Successfully deleted GitHub token for user {self.user_morphik_id}")
                self.client = None
                return True
            except Exception as e:
                logger.error(f"Error deleting GitHub token for user {self.user_morphik_id}: {e}")
                return False
        
        self.client = None
        return True

    async def pack_repository_smart_chunking(
        self,
        repo_path: str,
        chunk_by_type: bool = True,
        max_chunk_size: int = 100000
    ) -> List[Dict[str, Any]]:
        """
        Pack a GitHub repository into intelligent chunks categorized by file type.
        
        Args:
            repo_path: Repository path in format 'owner/repo'
            chunk_by_type: Whether to chunk by file type (True) or by size (False)
            max_chunk_size: Maximum size per chunk in characters
            
        Returns:
            List of chunks with metadata
        """
        if not self.client:
            raise ConnectionError("Not authenticated with GitHub")

        try:
            # Get repository
            parts = repo_path.split('/')
            if len(parts) != 2:
                raise ValueError("Invalid repository path. Expected format: 'owner/repo'")
            
            owner, repo_name = parts
            repo = self.client.get_repo(f"{owner}/{repo_name}")
            
            # Initialize chunk categories
            chunks = {
                "source": {"files": [], "content": "", "extensions": set()},
                "docs": {"files": [], "content": "", "extensions": set()},
                "config": {"files": [], "content": "", "extensions": set()},
                "tests": {"files": [], "content": "", "extensions": set()},
                "data": {"files": [], "content": "", "extensions": set()}
            }
            
            # File type categorization
            source_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.clj', '.hs', '.elm', '.ml', '.fs', '.vb', '.pas', '.ada', '.cob', '.pl', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd'}
            doc_extensions = {'.md', '.txt', '.rst', '.adoc', '.tex', '.pdf', '.doc', '.docx', '.rtf', '.html', '.htm', '.xml'}
            config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.properties', '.env', '.lock'}
            test_extensions = {'.test.', '.spec.', '_test.', '_spec.'}
            data_extensions = {'.csv', '.tsv', '.json', '.xml', '.sql', '.db', '.sqlite', '.xlsx', '.xls'}
            
            def categorize_file(path: str) -> str:
                """Categorize a file based on its path and extension."""
                path_lower = path.lower()
                
                # Check for test files first (by naming pattern)
                if any(pattern in path_lower for pattern in test_extensions):
                    return "tests"
                if 'test' in path_lower or 'spec' in path_lower:
                    return "tests"
                
                # Get file extension
                ext = Path(path).suffix.lower()
                
                # Categorize by extension
                if ext in source_extensions:
                    return "source"
                elif ext in doc_extensions:
                    return "docs"
                elif ext in config_extensions:
                    return "config"
                elif ext in data_extensions:
                    return "data"
                else:
                    # Default categorization by common file names
                    if any(name in path_lower for name in ['readme', 'license', 'changelog', 'contributing', 'authors', 'credits']):
                        return "docs"
                    elif any(name in path_lower for name in ['package.json', 'requirements.txt', 'pyproject.toml', 'pom.xml', 'build.gradle', 'makefile', 'dockerfile', 'docker-compose', '.gitignore', '.github']):
                        return "config"
                    else:
                        return "source"  # Default to source
            
            # Recursively fetch repository contents
            def get_all_files(contents):
                """Recursively get all files from repository contents."""
                files = []
                for item in contents:
                    if item.type == "file":
                        files.append(item)
                    elif item.type == "dir":
                        try:
                            sub_contents = repo.get_contents(item.path)
                            files.extend(get_all_files(sub_contents if isinstance(sub_contents, list) else [sub_contents]))
                        except Exception as e:
                            logger.warning(f"Could not access directory {item.path}: {e}")
                return files
            
            # Get all files in repository
            try:
                root_contents = repo.get_contents("")
                all_files = get_all_files(root_contents if isinstance(root_contents, list) else [root_contents])
            except Exception as e:
                logger.error(f"Could not fetch repository contents: {e}")
                raise
            
            # Process each file
            processed_files = 0
            for file_item in all_files:
                try:
                    # Skip binary files and very large files
                    if file_item.size > 1024 * 1024:  # Skip files larger than 1MB
                        continue
                    
                    # Get file content
                    file_content = file_item.decoded_content.decode('utf-8', errors='ignore')
                    category = categorize_file(file_item.path)
                    
                    # Add to appropriate chunk
                    chunks[category]["files"].append({
                        "path": file_item.path,
                        "size": file_item.size,
                        "sha": file_item.sha
                    })
                    chunks[category]["extensions"].add(Path(file_item.path).suffix.lower())
                    
                    # Escape XML special characters in file content
                    escaped_content = self._escape_xml_content(file_content)
                    escaped_path = self._escape_xml_content(file_item.path)
                    
                    # Add content with file header
                    chunks[category]["content"] += f"\n\n<!-- File: {escaped_path} -->\n"
                    chunks[category]["content"] += f"<file path=\"{escaped_path}\">\n{escaped_content}\n</file>"
                    
                    processed_files += 1
                    
                    # Limit number of files for very large repositories
                    if processed_files > 500:
                        logger.warning(f"Repository {repo_path} has many files, limiting to first 500 for performance")
                        break
                        
                except Exception as e:
                    logger.warning(f"Could not process file {file_item.path}: {e}")
                    continue
            
            # Build final chunks
            result_chunks = []
            chunk_id = 1
            
            for chunk_type, chunk_data in chunks.items():
                if chunk_data["files"]:  # Only include chunks with files
                    # Escape metadata values
                    escaped_repo_name = self._escape_xml_content(repo_name)
                    escaped_owner = self._escape_xml_content(owner)
                    escaped_extensions = self._escape_xml_content(', '.join(sorted(chunk_data["extensions"])))
                    
                    # Wrap content in XML structure
                    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<repository>
    <metadata>
        <name>{escaped_repo_name}</name>
        <owner>{escaped_owner}</owner>
        <type>{chunk_type}</type>
        <files_count>{len(chunk_data["files"])}</files_count>
        <extensions>{escaped_extensions}</extensions>
        <url>https://github.com/{repo_path}</url>
    </metadata>
    <files>
{chunk_data["content"]}
    </files>
</repository>"""
                    
                    result_chunks.append({
                        "id": chunk_id,
                        "type": chunk_type,
                        "content": xml_content,
                        "size": len(xml_content),
                        "files_count": len(chunk_data["files"]),
                        "extensions": list(chunk_data["extensions"])
                    })
                    chunk_id += 1
            
            logger.info(f"Packed repository {repo_path} into {len(result_chunks)} chunks with {processed_files} files")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Error packing repository {repo_path}: {e}")
            raise

    async def ingest_repository(
        self,
        repo_path: str,
        document_service: "DocumentService",
        auth_context,
        redis,
        folder_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        compress: bool = True,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ingest an entire GitHub repository as structured documents using Repomix-style packing.
        
        This creates multiple documents:
        1. Repository overview document (metadata, structure)
        2. Source code document (categorized by language/type)
        3. Documentation document (README, docs, etc.)
        4. Configuration document (config files, package.json, etc.)
        5. Tests document (test files)
        
        Args:
            repo_path: Repository path in format 'owner/repo'
            document_service: An initialized DocumentService instance.
            auth_context: Authentication context for ingestion.
            redis: Redis connection for queueing jobs.
            folder_name: Morphik folder to ingest into
            metadata: Additional metadata for documents
            include_patterns: Patterns for files to include
            ignore_patterns: Patterns for files to ignore  
            compress: Whether to use intelligent compression
            
        Returns:
            List of created document metadata
        """
        if not self.client:
            raise ConnectionError("Not authenticated with GitHub")

        try:
            # Get repository metadata first
            parts = repo_path.split('/')
            if len(parts) != 2:
                raise ValueError("Invalid repository path. Expected format: 'owner/repo'")
            
            owner, repo_name = parts
            repo = self.client.get_repo(f"{owner}/{repo_name}")
            
            # Check for existing repository documents to prevent duplicates
            if not force:
                try:
                    from core.database.postgres_database import PostgresDatabase
                    db = PostgresDatabase()
                    existing_docs = await db.search_documents(
                        user_id=auth_context.user_id,
                        query="",
                        metadata_filters={"repository": repo_path, "source": "github_repository"},
                        limit=10
                    )
                    
                    if existing_docs:
                        logger.warning(f"Repository {repo_path} already exists with {len(existing_docs)} documents. Skipping duplicate ingestion. Use force=true to re-ingest.")
                        # Return existing document info instead of creating duplicates
                        return [
                            {
                                "document_id": doc.external_id,
                                "filename": f"{repo_name}_{doc.system_metadata.get('chunk_type', 'unknown')}.xml",
                                "chunk_type": doc.system_metadata.get('chunk_type', 'unknown'),
                                "size": doc.system_metadata.get('chunk_size', 0),
                                "files_count": doc.system_metadata.get('files_count', 0),
                                "status": "already_exists"
                            }
                            for doc in existing_docs
                        ]
                except Exception as e:
                    logger.warning(f"Could not check for existing documents: {e}. Proceeding with ingestion.")
            else:
                logger.info(f"Force flag set to true. Re-ingesting repository {repo_path} even if it already exists.")
            
            # Pack repository using smart chunking approach
            chunks = await self.pack_repository_smart_chunking(
                repo_path=repo_path,
                chunk_by_type=True,
                max_chunk_size=100000  # Larger chunks for ingestion
            )
            
            # Prepare base metadata
            base_metadata = {
                "source": "github_repository",
                "repository": repo_path,
                "repository_url": f"https://github.com/{repo_path}",
                "stars": repo.stargazers_count,
                "language": repo.language,
                "description": repo.description,
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                **(metadata or {})
            }
            
            created_documents = []
            
            # Create documents for each chunk category
            for chunk in chunks:
                chunk_metadata = {
                    **base_metadata,
                    "chunk_type": chunk["type"],
                    "files_count": chunk["files_count"],
                    "chunk_size": chunk["size"],
                    "chunk_id": chunk["id"]
                }
                
                # Create document filename based on chunk type
                filename = f"{repo_name}_{chunk['type']}.xml"
                
                # Convert content to bytes
                content_bytes = chunk["content"].encode('utf-8')
                
                # Ingest using existing document service
                if auth_context and redis:
                    doc = await document_service.ingest_file_content(
                        file_content_bytes=content_bytes,
                        filename=filename,
                        content_type="application/xml",
                        metadata=chunk_metadata,
                        auth=auth_context,
                        redis=redis,
                        folder_name=folder_name,
                        rules=None,  # Could add repository-specific rules here
                        use_colpali=False  # XML content works better with standard embeddings
                    )
                    created_documents.append({
                        "document_id": doc.external_id,
                        "filename": filename,
                        "chunk_type": chunk["type"],
                        "size": chunk["size"],
                        "files_count": chunk["files_count"]
                    })
            
            logger.info(f"Successfully ingested repository {repo_path} as {len(created_documents)} documents")
            return created_documents
            
        except Exception as e:
            logger.error(f"Error ingesting GitHub repository {repo_path}: {e}")
            raise

    async def ingest_repository_simple(
        self,
        repo_path: str,
        folder_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        compress: bool = True,
        auth_context = None,
        redis = None
    ) -> Dict[str, Any]:
        """
        Ingest entire repository as a single document using Repomix CLI.
        This is faster but creates one large document instead of categorized chunks.
        """
        if not self.client:
            raise ConnectionError("Not authenticated with GitHub")

        try:
            # Pack repository using Repomix CLI
            packed_repo = await self._pack_remote_repository_with_repomix(
                repo_url=f"https://github.com/{repo_path}",
                include_patterns=include_patterns,
                ignore_patterns=ignore_patterns,
                compress=compress,
                style="xml"
            )
            
            # Prepare metadata
            repo_metadata = {
                "source": "github_repository_simple",
                "repository": repo_path,
                "repository_url": f"https://github.com/{repo_path}",
                "compressed": packed_repo["compressed"],
                "files_processed": packed_repo["metadata"]["files_processed"],
                "total_tokens": packed_repo["metadata"]["total_tokens"],
                **(metadata or {})
            }
            
            # Import document service
            from core.services.document_service import DocumentService
            
            document_service = DocumentService()
            
            # Create filename
            parts = repo_path.split('/')
            repo_name = parts[1]
            filename = f"{repo_name}_complete.xml"
            
            # Convert content to bytes
            content_bytes = packed_repo["content"].encode('utf-8')
            
            # Ingest using existing document service
            if auth_context and redis:
                doc = await document_service.ingest_file_content(
                    file_content_bytes=content_bytes,
                    filename=filename,
                    content_type="application/xml",
                    metadata=repo_metadata,
                    auth=auth_context,
                    redis=redis,
                    folder_name=folder_name,
                    rules=None,
                    use_colpali=False
                )
                
                return {
                    "document_id": doc.external_id,
                    "filename": filename,
                    "repository": repo_path,
                    "size_bytes": packed_repo["size_bytes"],
                    "files_processed": packed_repo["metadata"]["files_processed"],
                    "total_tokens": packed_repo["metadata"]["total_tokens"]
                }
            
        except Exception as e:
            logger.error(f"Error in simple repository ingestion for {repo_path}: {e}")
            raise 