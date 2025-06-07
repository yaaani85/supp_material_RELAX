from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import requests
from urllib.parse import quote
from pathlib import Path
from requests.exceptions import RequestException, Timeout
from query_evaluation.dataset import QEDataset
import logging
import psutil
import subprocess
import time

class TripleStoreException(Exception):
    """Custom exception for TripleStore-related errors."""
    def __init__(self, message: str, is_timeout: bool = False):
        super().__init__(message)
        self.is_timeout = is_timeout

class TripleStore:
    """A class to interact with a GraphDB triple store.
    
    Handles repository management, data loading, and query execution for RDF data.
    
    Attributes:
        endpoint: Default Base URL of local GraphDB instances
        repository_id: GraphDB ID of the repository to use
        dataset: QEDataset instance containing the data to load
    """
    
    MIN_TRIPLES_THRESHOLD = 70  # GraphDB inits with 
    
    def __init__(
        self, 
        repository_id: str = None, 
        dataset: QEDataset = None, 
        endpoint: str = None,
    ):
        # Load environment variables
        load_dotenv()
        
        self.query_count = 0  # Track number of queries executed
        self.endpoint = endpoint or os.getenv('GRAPHDB_URL', 'http://localhost:7200').rstrip('/')
        self.repository_id = repository_id or os.getenv('GRAPHDB_REPOSITORY_ID')
        self.default_timeout = int(os.getenv('GRAPHDB_TIMEOUT', '10'))
        self.graphdb_path = os.getenv('GRAPHDB_PATH')
        
        if not self.graphdb_path:
            raise TripleStoreException("GRAPHDB_PATH must be set in environment variables")
        
        if not self.repository_id:
            raise TripleStoreException("Repository ID must be provided either through constructor or GRAPHDB_REPOSITORY_ID env variable")
            
        self.repository_endpoint = f"{self.endpoint}/repositories/{self.repository_id}"
        self.dataset = dataset
        self.headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Health check before proceeding
        # self._health_check()
        self.ensure_server_running()
        
        # Initialize repository if needed
        self._ensure_repository_exists()
        self._ensure_data_loaded()
        
    
    def _ensure_repository_exists(self) -> None:
        """Check if repository exists, raise exception if it doesn't."""
        try:
            repos_url = f"{self.endpoint}/repositories"
            response = requests.get(repos_url, headers=self.headers)
            response.raise_for_status()
            repositories = response.json()
            
            repo_exists = any(
                binding['id']['value'] == self.repository_id 
                for binding in repositories['results']['bindings']
            )
            
            if not repo_exists:
                raise TripleStoreException(f"Repository {self.repository_id} does not exist")
            logging.info(f"Repository {self.repository_id} exists")
            
        except (RequestException, Timeout) as e:
            raise TripleStoreException(f"Failed to check repository: {str(e)}")

    def _ensure_data_loaded(self) -> None:
        """Check if dataset is loaded, upload if not."""
        try:
            # Simpler query that returns as soon as it finds a single triple
            check_query = "ASK WHERE { ?s ?p ?o } LIMIT 1"
            result = self.execute_query(check_query)
            
            if not result.get('boolean', False):
                logging.info("Loading dataset into triple store")
                self._load_dataset()
            else:
                logging.info("Dataset already loaded (found existing triples)")
                
        except TripleStoreException as e:
            if e.is_timeout:
                logging.warning(f"Timeout while checking data load status. Assuming data is loaded: {str(e)}")
                return
            else:
                raise TripleStoreException(f"Failed to check data loading status: {str(e)}")

    def execute_query(self, query: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Executes a SPARQL query against the configured endpoint."""
        self.query_count += 1
        current_query_id = self.query_count
        
        try:
            encoded_query = quote(query)
            response = requests.get(
                f"{self.repository_endpoint}?query={encoded_query}",
                headers=self.headers,
                timeout=timeout or self.default_timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Timeout:
            logging.error(
                f"Query timed out after {timeout or self.default_timeout}s:\n"
                f"Query ID: {current_query_id}\n"
                f"Query: {query}\n"
            )
            # Force restart on timeout
            self.ensure_server_running(force_restart=True)
            # If we get here, server restarted successfully, but we still want to mark this query as timeout
            raise TripleStoreException(f"Query {current_query_id} timed out", is_timeout=True)
            
        except (RequestException) as e:
            logging.error(
                f"Query execution failed:\n"
                f"Query ID: {current_query_id}\n"
                f"Query: {query}\n"
                f"Error: {str(e)}\n"
            )
            raise TripleStoreException(f"Query execution failed: {str(e)}", is_timeout=False)

    def _load_dataset(self) -> None:
        """Load the dataset's (train+validation) .nt files into the triple store."""
        files_to_load = [
            self.dataset.train_split_location(),
            self.dataset.validation_split_location(),
        ]
        
        for file_path in files_to_load:
            if Path(file_path).exists():
                self._upload_file(Path(file_path))
            else:
                raise Exception(f"File {file_path} does not exist")
    
    def _upload_file(self, file_path: Path) -> None:
        """Upload a single .nt file to the triple store using the SPARQL/RDF4J API.
        
        Args:
            file_path: Path to the .nt file to upload
        """
        upload_url = f"{self.repository_endpoint}/statements"
        headers = {'Content-Type': "application/n-triples"}
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                upload_url,
                data=f,
                headers=headers
            )
            response.raise_for_status()
            logging.info(f"Uploaded {file_path.name}")

    def _health_check(self) -> None:
        """Verify GraphDB is running and responsive."""
        try:
            repos_url = f"{self.endpoint}/repositories"
            response = requests.get(repos_url, headers=self.headers, timeout=5)
            response.raise_for_status()
            logging.info(f"GraphDB is running at {self.endpoint}")
        except (RequestException, Timeout) as e:
            raise TripleStoreException(
                f"Health check failed: GraphDB appears to be down or unreachable at {self.endpoint}. "
                f"Error: {str(e)}"
            )

    def ensure_server_running(self, force_restart: bool = False) -> None:
        """Ensure GraphDB server is running, start or restart if needed."""
        logging.info("Checking if GraphDB is running...")
        
        try:
            self._health_check()
            if force_restart:
                logging.info("Force restart requested, restarting GraphDB...")
            else:
                logging.info("GraphDB is already running")
                return
        except TripleStoreException:
            logging.info("GraphDB not running or unhealthy, attempting to start...")
            
        try:
            # Kill any process using port 7200 more aggressively
            for _ in range(3):  # Try multiple times
                # Find processes using port 7200
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        connections = proc.connections()
                        for conn in connections:
                            if conn.laddr.port == 7200:
                                logging.info(f"Killing process {proc.pid} using port 7200")
                                proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                time.sleep(2)  # Short wait between kill attempts
                
                # Check if port is still in use
                in_use = False
                for proc in psutil.process_iter(['connections']):
                    try:
                        connections = proc.connections()
                        for conn in connections:
                            if conn.laddr.port == 7200:
                                in_use = True
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if not in_use:
                    break
            
            if in_use:
                raise TripleStoreException("Failed to free port 7200 after multiple kill attempts")
                
            logging.info("Port 7200 is free")
            
            # Start GraphDB in server-only and daemon mode using path from env
            cmd = [self.graphdb_path, '-s', '-d']
            logging.info(f"Starting GraphDB with command: {' '.join(cmd)}")
            subprocess.Popen(cmd,
                            start_new_session=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            
            # Wait for server to be ready
            max_attempts = 12  # 60 seconds total
            for attempt in range(max_attempts):
                try:
                    time.sleep(5)
                    self._health_check()
                    logging.info("GraphDB server successfully started!")
                    return
                except TripleStoreException:
                    if attempt == max_attempts - 1:
                        raise TripleStoreException("Failed to start GraphDB server")
                    logging.info(f"Waiting for GraphDB to start... (attempt {attempt + 1}/{max_attempts})")
                    continue
                    
        except Exception as e:
            raise TripleStoreException(f"Failed to start GraphDB server: {str(e)}")