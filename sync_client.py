import requests
import logging
import json
import time
from typing import List, Dict, Optional

class CloudSync:
    """
    Client for synchronizing telemetry events with the Transport Admin API.
    Handles authentication, batching, and error recovery.
    """
    
    def __init__(self, api_url: str, api_token: str, device_mac: str):
        """
        Initialize the sync client.
        
        Args:
            api_url: Base URL of the API (e.g., http://localhost:8000/api/v1)
            api_token: Bearer token for authentication
            device_mac: MAC address of the device (used for logging/identification)
        """
        self.api_url = api_url.rstrip('/')
        self.api_token = api_token
        self.device_mac = device_mac
        self.logger = logging.getLogger(__name__)
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Device-MAC': self.device_mac
        }

    def check_connection(self) -> bool:
        """Check if the API is reachable."""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            # Accept 200 or 401 (unauthorized but reachable) as "connected" for status check
            return response.status_code in [200, 401]
        except requests.RequestException:
            return False

    def authenticate_device(self) -> dict:
        """
        Authenticate with the backend using the device MAC address.
        The endpoint always returns the device token on success.
        
        Returns:
            dict with keys:
                - status: 'authenticated', 'pending', 'registered', 'inactive', 'error'
                - message: Human-readable message
                - token: API token (always present on success)
                - device: Device info dict (if available)
        """
        auth_url = f"{self.api_url}/device/auth"
        
        try:
            self.logger.info(f"Authenticating device with MAC: {self.device_mac}")
            response = requests.post(
                auth_url,
                json={'mac_address': self.device_mac},
                headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
                timeout=10
            )
            
            data = response.json()
            status = data.get('status', 'error')
            token = data.get('token')
            
            # Always capture token if present
            if token:
                self.api_token = token
                self.headers['Authorization'] = f'Bearer {self.api_token}'
            
            if response.status_code == 200 and data.get('success'):
                self.logger.info(f"Device auth response: {status}")
                return {
                    'status': status,
                    'message': data.get('message', ''),
                    'token': token,
                    'device': data.get('device', {})
                }
            
            if response.status_code == 403:
                self.logger.warning(f"Device inactive: {data.get('message')}")
                return {
                    'status': 'inactive',
                    'message': data.get('message', 'Dispositivo inactivo')
                }
            
            self.logger.error(f"Auth failed: {response.status_code} - {response.text}")
            return {
                'status': 'error',
                'message': f"Error {response.status_code}: {data.get('message', 'Unknown error')}"
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Network error during authentication: {e}")
            return {
                'status': 'error',
                'message': f"Error de red: {e}"
            }

    def sync_events(self, events: List[Dict]) -> int:
        """
        Sync a batch of events to the cloud.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Number of successfully synced events. Returns 0 on failure.
        """
        if not events:
            return 0
            
        payload = {
            "events": events
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/sync",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                synced_count = result.get('synced_count', len(events))
                self.logger.info(f"Successfully synced {synced_count} events")
                return synced_count
            else:
                self.logger.error(f"Sync failed with status {response.status_code}: {response.text}")
                return 0
                
        except requests.RequestException as e:
            self.logger.error(f"Network error during sync: {e}")
            return 0

    def send_heartbeat(self) -> bool:
        """
        Send a heartbeat payload to the API to update device status.
        
        Returns:
            True if heartbeat was successful
        """
        try:
            payload = {"events": []}
            response = requests.post(
                f"{self.api_url}/sync",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.debug("Heartbeat sent successfully")
                return True
            else:
                self.logger.warning(f"Heartbeat failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"Network error during heartbeat: {e}")
            return False

    def get_excluded_faces(self) -> List[Dict]:
        """
        Obtiene lista de conductores con foto para excluir del conteo.
        """
        try:
            self.logger.info("Sincronizando rostros excluidos...")
            response = requests.get(
                f"{self.api_url}/drivers/excluded",
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    drivers = data.get("drivers", [])
                    self.logger.info(f"Se obtuvieron {len(drivers)} conductores para exclusión")
                    return drivers
            
            self.logger.warning(f"Error obteniendo rostros excluidos: {response.status_code} - {response.text}")
            return []
            
        except Exception as e:
            self.logger.error(f"Excepción obteniendo rostros excluidos: {e}")
            return []

    def get_excluded_collectors(self) -> List[Dict]:
        """
        Obtiene lista de colectores con foto para excluir del conteo.
        """
        try:
            self.logger.info("Sincronizando colectores excluidos...")
            response = requests.get(
                f"{self.api_url}/collectors/excluded",
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    collectors = data.get("collectors", [])
                    self.logger.info(f"Se obtuvieron {len(collectors)} colectores para exclusión")
                    return collectors
            
            self.logger.warning(f"Error obteniendo colectores excluidos: {response.status_code} - {response.text}")
            return []
            
        except Exception as e:
            self.logger.error(f"Excepción obteniendo colectores excluidos: {e}")
            return []
