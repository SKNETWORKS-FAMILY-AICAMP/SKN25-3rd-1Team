import os
import json
from celery import Celery
from pymongo import MongoClient
from redis import Redis
from dotenv import load_dotenv

load_dotenv()

# Celery 및 Redis 설정 (Broker로 Redis 사용)
CELERY_BROKER = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/0"
celery_app = Celery("log_worker", broker=CELERY_BROKER)

# MongoDB 및 Redis 클라이언트 초기화
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["chatbot_db"]
redis_client = Redis(
    host=os.getenv("REDIS_HOST"), 
    port=int(os.getenv("REDIS_PORT")), 
    decode_responses=True
)

# 5초마다 주기적으로 스트림 플러시 작업 예약
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(5.0, flush_streams_to_mongo.s(), name='flush_logs_every_5s')

@celery_app.task
def flush_streams_to_mongo():
    streams = {
        "stream:usage_logs": "0-0", 
        "stream:node_perf": "0-0"
    }
    
    # 각 스트림에서 최대 100개씩 읽어옴
    messages = redis_client.xread(streams, count=100)
    
    for stream_name, msgs in messages:
        if not msgs:
            continue
            
        docs = []
        msg_ids = []
        
        for msg_id, data in msgs:
            docs.append(json.loads(data["payload"]))
            msg_ids.append(msg_id)
            
        if docs:
            # MongoDB에 일괄 삽입
            if stream_name == "stream:usage_logs":
                db["usage_logs"].insert_many(docs)
            elif stream_name == "stream:node_perf":
                db["node_perf_logs"].insert_many(docs)
            
            # 처리 완료된 메시지는 스트림에서 삭제
            #redis_client.xdel(stream_name, *msg_ids)