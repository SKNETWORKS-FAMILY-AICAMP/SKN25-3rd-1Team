import os
import json
from datetime import datetime
from redis import Redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))

redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT,db=0, decode_responses=True)

def save_usage_log(log_data: dict):
    try:
        # Redis Streams에 데이터 추가
        redis_client.xadd("stream:usage_logs", {"payload": json.dumps(log_data)})
    except Exception as e:
        print(f"[Redis Error] stream:usage_logs 저장 실패: {e}")

def save_node_perf(trace_id: str, node_name: str, duration: float, metadata: dict = None):
    try:
        log_entry = {
            "trace_id": trace_id,
            "node_name": node_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            log_entry["metadata"] = json.dumps(metadata)
            
        # Redis Streams에 데이터 추가
        redis_client.xadd("stream:node_perf", {"payload": json.dumps(log_entry)})
    except Exception as e:
        print(f"[Redis Error] stream:node_perf 저장 실패: {e}")