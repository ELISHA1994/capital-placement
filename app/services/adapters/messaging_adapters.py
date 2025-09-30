"""
Local messaging implementations for development.
"""

import asyncio
import json
import uuid
from collections import deque
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import structlog

from app.core.interfaces import IMessageQueue, IEventPublisher, Message

logger = structlog.get_logger(__name__)

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class InMemoryMessageQueue(IMessageQueue):
    """In-memory message queue for local development and testing."""
    
    def __init__(self):
        self._queues: Dict[str, deque] = {}
        self._lock = threading.RLock()
    
    def _ensure_queue(self, queue_name: str):
        """Ensure queue exists."""
        with self._lock:
            if queue_name not in self._queues:
                self._queues[queue_name] = deque()
    
    async def send_message(self, queue_name: str, message: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """Send message to in-memory queue."""
        try:
            message_id = str(uuid.uuid4())
            
            message_obj = Message(
                id=message_id,
                body=message,
                properties=properties or {},
                delivery_count=0
            )
            
            with self._lock:
                self._ensure_queue(queue_name)
                self._queues[queue_name].append(message_obj)
            
            logger.debug("Message sent to in-memory queue",
                        queue_name=queue_name,
                        message_id=message_id)
            
            return message_id
        except Exception as e:
            logger.error("Failed to send message to in-memory queue",
                        queue_name=queue_name,
                        error=str(e))
            raise
    
    async def receive_messages(self, queue_name: str, max_messages: int = 1) -> List[Message]:
        """Receive messages from in-memory queue."""
        try:
            messages = []
            
            with self._lock:
                self._ensure_queue(queue_name)
                queue = self._queues[queue_name]
                
                for _ in range(min(max_messages, len(queue))):
                    if queue:
                        message = queue.popleft()
                        message.delivery_count += 1
                        messages.append(message)
            
            logger.debug("Messages received from in-memory queue",
                        queue_name=queue_name,
                        count=len(messages))
            
            return messages
        except Exception as e:
            logger.error("Failed to receive messages from in-memory queue",
                        queue_name=queue_name,
                        error=str(e))
            raise
    
    async def complete_message(self, queue_name: str, message: Message) -> bool:
        """Mark message as completed (no-op for in-memory)."""
        logger.debug("Message completed (in-memory)",
                    queue_name=queue_name,
                    message_id=message.id)
        return True
    
    async def abandon_message(self, queue_name: str, message: Message) -> bool:
        """Abandon message (put back in queue)."""
        try:
            with self._lock:
                self._ensure_queue(queue_name)
                self._queues[queue_name].appendleft(message)
            
            logger.debug("Message abandoned (back to queue)",
                        queue_name=queue_name,
                        message_id=message.id)
            return True
        except Exception as e:
            logger.error("Failed to abandon message",
                        queue_name=queue_name,
                        message_id=message.id,
                        error=str(e))
            return False
    
    async def dead_letter_message(self, queue_name: str, message: Message) -> bool:
        """Send message to dead letter queue."""
        try:
            dead_letter_queue = f"{queue_name}_deadletter"
            await self.send_message(dead_letter_queue, message.body, message.properties)
            
            logger.debug("Message sent to dead letter queue",
                        queue_name=queue_name,
                        dead_letter_queue=dead_letter_queue,
                        message_id=message.id)
            return True
        except Exception as e:
            logger.error("Failed to send message to dead letter",
                        queue_name=queue_name,
                        message_id=message.id,
                        error=str(e))
            return False


class RedisMessageQueue(IMessageQueue):
    """Redis-based message queue for local development."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        self.redis_url = redis_url
        self._redis = None
    
    async def _get_redis(self):
        """Get Redis connection with lazy initialization."""
        if self._redis is None:
            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self._redis.ping()
                logger.info("Redis message queue connection established")
            except Exception as e:
                logger.error("Failed to connect to Redis for messaging", error=str(e))
                raise
        
        return self._redis
    
    async def send_message(self, queue_name: str, message: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """Send message to Redis queue."""
        try:
            redis_client = await self._get_redis()
            
            message_id = str(uuid.uuid4())
            message_obj = {
                "id": message_id,
                "body": message,
                "properties": properties or {},
                "delivery_count": 0,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Use Redis list as queue
            await redis_client.lpush(f"queue:{queue_name}", json.dumps(message_obj))
            
            logger.debug("Message sent to Redis queue",
                        queue_name=queue_name,
                        message_id=message_id)
            
            return message_id
        except Exception as e:
            logger.error("Failed to send message to Redis queue",
                        queue_name=queue_name,
                        error=str(e))
            raise
    
    async def receive_messages(self, queue_name: str, max_messages: int = 1) -> List[Message]:
        """Receive messages from Redis queue."""
        try:
            redis_client = await self._get_redis()
            messages = []
            
            for _ in range(max_messages):
                # Use BRPOP for blocking pop (with timeout)
                result = await redis_client.brpop(f"queue:{queue_name}", timeout=0.1)
                if not result:
                    break
                
                _, message_data = result
                message_obj = json.loads(message_data)
                
                message = Message(
                    id=message_obj["id"],
                    body=message_obj["body"],
                    properties=message_obj.get("properties", {}),
                    delivery_count=message_obj.get("delivery_count", 0)
                )
                
                message.delivery_count += 1
                messages.append(message)
            
            logger.debug("Messages received from Redis queue",
                        queue_name=queue_name,
                        count=len(messages))
            
            return messages
        except Exception as e:
            logger.error("Failed to receive messages from Redis queue",
                        queue_name=queue_name,
                        error=str(e))
            raise
    
    async def complete_message(self, queue_name: str, message: Message) -> bool:
        """Mark message as completed."""
        logger.debug("Message completed (Redis)",
                    queue_name=queue_name,
                    message_id=message.id)
        return True
    
    async def abandon_message(self, queue_name: str, message: Message) -> bool:
        """Abandon message (put back in queue)."""
        try:
            redis_client = await self._get_redis()
            
            message_obj = {
                "id": message.id,
                "body": message.body,
                "properties": message.properties,
                "delivery_count": message.delivery_count
            }
            
            await redis_client.lpush(f"queue:{queue_name}", json.dumps(message_obj))
            
            logger.debug("Message abandoned (back to Redis queue)",
                        queue_name=queue_name,
                        message_id=message.id)
            return True
        except Exception as e:
            logger.error("Failed to abandon message to Redis queue",
                        queue_name=queue_name,
                        message_id=message.id,
                        error=str(e))
            return False
    
    async def dead_letter_message(self, queue_name: str, message: Message) -> bool:
        """Send message to dead letter queue."""
        try:
            dead_letter_queue = f"{queue_name}_deadletter"
            await self.send_message(dead_letter_queue, message.body, message.properties)
            
            logger.debug("Message sent to Redis dead letter queue",
                        queue_name=queue_name,
                        dead_letter_queue=dead_letter_queue,
                        message_id=message.id)
            return True
        except Exception as e:
            logger.error("Failed to send message to Redis dead letter",
                        queue_name=queue_name,
                        message_id=message.id,
                        error=str(e))
            return False


class LocalEventPublisher(IEventPublisher):
    """Local event publisher for development."""
    
    def __init__(self):
        self._subscribers: Dict[str, List] = {}
        self._lock = threading.RLock()
    
    async def publish_event(self, topic: str, event_data: Dict[str, Any]) -> bool:
        """Publish event locally (just log for now)."""
        try:
            event_id = str(uuid.uuid4())
            
            logger.info("Event published locally",
                       topic=topic,
                       event_id=event_id,
                       event_data=event_data)
            
            return True
        except Exception as e:
            logger.error("Failed to publish event locally",
                        topic=topic,
                        error=str(e))
            return False
    
    async def publish_events(self, topic: str, events: List[Dict[str, Any]]) -> bool:
        """Publish multiple events locally."""
        try:
            success_count = 0
            
            for event_data in events:
                if await self.publish_event(topic, event_data):
                    success_count += 1
            
            logger.info("Events published locally",
                       topic=topic,
                       total=len(events),
                       successful=success_count)
            
            return success_count == len(events)
        except Exception as e:
            logger.error("Failed to publish events locally",
                        topic=topic,
                        error=str(e))
            return False