import http.server
import socketserver
import threading
import json
import time
import urllib.request
import urllib.parse
import sys

# --- CONFIG ---
ROUTER_PORT = 8010
SHARD_PORTS = [8011, 8012, 8013]
TOPICS = ["Science", "History", "Coding"]

# --- MOCK KNOWLEDGE ---
KNOWLEDGE = {
    "Science": "The speed of light is 299,792,458 m/s.",
    "History": "Rome fell in 476 AD.",
    "Coding": "Python uses indentation."
}

# --- REUSABLE SERVER ---
class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

# --- SHARD SERVER ---
class ShardHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        body = json.loads(self.rfile.read(content_len))
        query = body["query_text"]
        
        my_topic = self.server.topic
        
        score = 0.0
        if my_topic.lower() in query.lower():
            score = 1.0
        elif "light" in query.lower() and my_topic == "Science": score = 0.9 
        elif "rome" in query.lower() and my_topic == "History": score = 0.9
        elif "python" in query.lower() and my_topic == "Coding": score = 0.9
            
        response = {
            "shard_id": f"Shard_{my_topic}",
            "best_text": KNOWLEDGE[my_topic],
            "score": score
        }
        
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args): return # Silence logs

def run_shard(port, topic):
    print(f"🐝 [Shard] Starting {topic} on port {port}...")
    server = ReuseTCPServer(("localhost", port), ShardHandler)
    server.topic = topic
    server.serve_forever()

# --- ROUTER SERVER ---
class RouterHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_len = int(self.headers.get('Content-Length'))
        body = json.loads(self.rfile.read(content_len))
        query = body["messages"][-1]["content"]
        
        print(f"👑 [Router] Broadcasting: '{query}'")
        
        # Broadcast to Shards
        results = []
        for port in SHARD_PORTS:
            try:
                req = urllib.request.Request(
                    f"http://localhost:{port}", 
                    data=json.dumps({"query_text": query}).encode(),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req) as f:
                    results.append(json.loads(f.read().decode()))
            except:
                pass
        
        # Consensus
        if not results:
            self.send_error(500, "No shards reachable")
            return

        winner = max(results, key=lambda x: x["score"])
        
        # Response
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": f"[Truth from {winner['shard_id']}] {winner['best_text']}"
                }
            }]
        }
        
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format, *args): return # Silence logs

def run_router():
    print(f"👑 [Router] Starting on port {ROUTER_PORT}...")
    server = ReuseTCPServer(("localhost", ROUTER_PORT), RouterHandler)
    server.serve_forever()

# --- MAIN ---
if __name__ == "__main__":
    threads = []
    
    # Start Shards
    for i, port in enumerate(SHARD_PORTS):
        t = threading.Thread(target=run_shard, args=(port, TOPICS[i]), daemon=True)
        t.start()
        threads.append(t)
        
    # Start Router
    t = threading.Thread(target=run_router, daemon=True)
    t.start()
    threads.append(t)
    
    time.sleep(2) # Warmup
    
    print("\n⚡ STARTING INTEGRATION TEST (Native HTTP) ⚡")
    
    test_q = "Tell me about Python coding."
    print(f"\nQUERY: {test_q}")
    
    req = urllib.request.Request(
        f"http://localhost:{ROUTER_PORT}/v1/chat/completions",
        data=json.dumps({"messages": [{"content": test_q}]}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as f:
            res = json.loads(f.read().decode())
            print(f"RESPONSE: {res['choices'][0]['message']['content']}")
            if "Shard_Coding" in res['choices'][0]['message']['content']:
                print("✅ SUCCESS: Router correctly selected Coding Shard.")
            else:
                print("❌ FAIL: Wrong shard selected.")
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        
    # Test 2
    test_q2 = "What happened in Rome?"
    print(f"\nQUERY: {test_q2}")
    req2 = urllib.request.Request(
        f"http://localhost:{ROUTER_PORT}/v1/chat/completions",
        data=json.dumps({"messages": [{"content": test_q2}]}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req2) as f:
        res = json.loads(f.read().decode())
        print(f"RESPONSE: {res['choices'][0]['message']['content']}")
        
    print("\n🛑 Test Complete.")