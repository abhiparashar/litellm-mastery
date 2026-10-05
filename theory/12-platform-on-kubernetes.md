# 12 — Running LiteLLM as a company platform on Kubernetes

Project: L1 Company AI platform (ROADMAP.md) · Read: before you start building

Previous: [11 — RAG and guardrails](./11-rag-and-guardrails.md) · Next: [13 — Smart routing and hooks](./13-smart-routing-and-hooks.md)

---

## The idea in plain words

Think of a busy bank branch.

- One cashier cannot serve 500 customers. So the bank opens **several counters** that all do the same job.
- At the door stands a **greeter** who sends each customer to a free counter.
- All counters read from **one shared ledger**. If counter 1 opens your account, counter 2 can see it.
- All counters share **one tally board** for limits ("this customer may withdraw 4 times today"). If each counter kept its own tally, you could withdraw 4 times at each counter.
- A **manager** watches the floor. If a cashier faints, the manager replaces them. If the queue grows, the manager opens more counters.

Now map it to LiteLLM:

| Bank | Your AI platform |
|---|---|
| Counter | One LiteLLM proxy copy (a **replica**) |
| Greeter at the door | **Load balancer**: a program that spreads requests over replicas |
| Shared ledger | **Postgres** database: virtual keys, teams, budgets, spend logs |
| Shared tally board | **Redis**: rate-limit counters, router state, cache |
| Manager | **Kubernetes**: starts, restarts, and scales the replicas |

A **platform team** is the group inside a company that builds this kind of shared service once, so other teams do not each build their own. Product teams get one URL and one key. The platform team owns uptime, cost limits, upgrades, and dashboards. In chapters 8–10 you built one gateway on your laptop. This chapter is about running it the way a platform team does: many copies, shared state, health checks, autoscaling, and proof by load test.

## Jargon table

| Term | Full name | What it does | Tiny example |
|---|---|---|---|
| Replica | – | One running copy of the same program | `proxy1`, `proxy2` |
| Load balancer | – | Spreads incoming requests over replicas | nginx `upstream` block |
| Stateless | – | Keeps nothing important in its own memory; any copy can serve any request | LiteLLM proxy with Postgres + Redis |
| Container | – | A packaged program plus everything it needs to run | `ghcr.io/berriai/litellm:v1.104.0` |
| Kubernetes (k8s) | – | A system that runs containers on many machines and keeps them healthy | "keep 3 proxies running" |
| Pod | – | The smallest thing Kubernetes runs: one or more containers together | one LiteLLM container |
| Deployment | – | Tells Kubernetes "keep N identical pods running" and handles upgrades | `replicas: 3` |
| Service | – | A stable name and address in front of changing pods | `ai-gw-litellm:4000` |
| ConfigMap | Configuration map | Stores non-secret config files for pods | LiteLLM `config.yaml` |
| Secret | – | Stores passwords and keys for pods (kept apart from config) | master key, `OPENAI_API_KEY` |
| HPA | Horizontal Pod Autoscaler | Adds or removes pods based on a metric | scale 3→10 when CPU > 70% |
| Probe | – | A check Kubernetes runs against a pod | GET `/health/readiness` |
| Liveness probe | – | "Is the process stuck?" Failing = restart the pod | `/health/liveliness` |
| Readiness probe | – | "May this pod get traffic now?" Failing = remove from load balancer | `/health/readiness` |
| Helm | – | Package manager for Kubernetes; fills templates from a values file | `helm install ai-gw oci://...` |
| Chart | Helm chart | A folder of Kubernetes templates plus default values | `litellm-helm` |
| Worker | – | One server process inside a container | `--num_workers 2` |
| RPS | Requests per second | Throughput | 366 req/s |
| p95 | 95th percentile | 95% of requests were faster than this | p95 = 500 ms |
| Load test | – | Fake many users to see where the system breaks | Locust, 200 users |

## The architecture

```mermaid
flowchart LR
    Apps[Team apps and agents] --> LB[Load balancer / Ingress]
    LB --> P1[LiteLLM pod 1]
    LB --> P2[LiteLLM pod 2]
    LB --> P3[LiteLLM pod N]
    P1 --> PG[(Postgres<br/>keys, teams, spend)]
    P2 --> PG
    P3 --> PG
    P1 --> R[(Redis<br/>rate limits, router state, cache)]
    P2 --> R
    P3 --> R
    P1 --> Prov[Providers<br/>OpenAI, Anthropic, Bedrock, Ollama]
    P2 --> Prov
    P3 --> Prov
    P1 -. /metrics .-> Prom[Prometheus] --> Graf[Grafana dashboards]
    HPA[HPA] -. adds or removes pods .-> P3
```

Why the shared stores matter:

- **Postgres** holds the virtual keys. A key created through pod 1 must work on pod 2. Without one shared database, each pod has its own world.
- **Redis** holds counters that change on every request: rate limits, spend counters, the Router's cooldown list and per-deployment usage. Postgres is too slow to update on every request, and pod memory is private. The LiteLLM production guide says to run Redis "as soon as you run more than one proxy instance", because without it "each instance enforces limits independently" ([docs.litellm.ai/docs/proxy/prod](https://docs.litellm.ai/docs/proxy/prod)).

You will prove both points with real requests below.

## The local lab: 2 replicas behind nginx

You do not need a cluster to learn the important ideas. Docker Compose can run the same shape on a laptop: nginx on port 4012, two proxy replicas, Redis on 6312, Postgres on 5412, and a mock model so no key is needed.

`config.yaml` (shared by both replicas):

```yaml
model_list:
  - model_name: fast-model
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: fake-key                 # never used: mock_response short-circuits the call
      mock_response: "Hello from the mock model"

router_settings:                        # Router state: cooldowns, per-deployment rpm/tpm
  redis_host: redis                     # every replica points at the SAME Redis
  redis_port: 6379

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL # every replica points at the SAME Postgres
  coordination_redis:                   # key/team rate limits + spend counters across pods
    host: redis
    port: 6379
```

Two Redis settings, two jobs. `router_settings.redis_*` shares Router state. `general_settings.coordination_redis` (verified in `litellm/proxy/_types.py`, 1.104.0) shares key and team rate limits and spend counters between pods.

`nginx.conf`:

```nginx
events {}

http {
  upstream litellm_replicas {
    server proxy1:4000;                 # replica 1
    server proxy2:4000;                 # replica 2
  }

  server {
    listen 80;

    location / {
      proxy_pass http://litellm_replicas;
      add_header X-Served-By $upstream_addr always;   # shows which replica answered
    }
  }
}
```

`docker-compose.yml` (the `${...:-default}` parts let you swap files from the shell later):

```yaml
name: lm-ch12

x-proxy: &proxy
  image: ghcr.io/berriai/litellm:v1.104.0
  command: ["--config", "/app/config.yaml", "--port", "4000"]
  volumes:
    - ${PROXY_CONFIG:-./config.yaml}:/app/config.yaml:ro
  env_file: .env                        # LITELLM_MASTER_KEY lives here, not in code
  environment:
    DATABASE_URL: postgresql://llm:llm@postgres:5432/litellm
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_started

services:
  postgres:
    image: postgres:16-alpine
    container_name: lm-ch12-postgres
    environment:
      POSTGRES_USER: llm
      POSTGRES_PASSWORD: llm
      POSTGRES_DB: litellm
    ports: ["5412:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U llm -d litellm"]
      interval: 2s
      retries: 30

  redis:
    image: redis:7-alpine
    container_name: lm-ch12-redis
    ports: ["6312:6379"]

  proxy1:
    <<: *proxy
    container_name: lm-ch12-proxy1

  proxy2:
    <<: *proxy
    container_name: lm-ch12-proxy2

  nginx:
    image: nginx:alpine
    container_name: lm-ch12-nginx
    volumes:
      - ${NGINX_CONF:-./nginx.conf}:/etc/nginx/nginx.conf:ro
    ports: ["4012:80"]
    depends_on: [proxy1, proxy2]
```

Start it with `docker compose up -d`. The `.env` file holds one line: `LITELLM_MASTER_KEY=sk-...`.

### Part 1: health endpoints through the load balancer

LiteLLM has public health endpoints made for probes. In 1.104.0 the source registers `/health/liveliness` (the historical spelling), `/health/liveness` (the Kubernetes spelling, same handler), and `/health/readiness`.

```python
import httpx

GATEWAY = "http://localhost:4012"   # nginx, in front of 2 proxy replicas

paths = ["/health/liveliness", "/health/liveness", "/health/readiness"]

for path in paths:
    for attempt in range(2):        # call twice so nginx picks both replicas
        response = httpx.get(GATEWAY + path)
        replica = response.headers.get("X-Served-By")
        print(path, response.status_code, replica, response.text)
```

Output (real, captured with litellm 1.104.0):

```text
/health/liveliness 200 172.20.0.4:4000 "I'm alive!"
/health/liveliness 200 172.20.0.5:4000 "I'm alive!"
/health/liveness 200 172.20.0.4:4000 "I'm alive!"
/health/liveness 200 172.20.0.5:4000 "I'm alive!"
/health/readiness 200 172.20.0.4:4000 {"status":"healthy","db":"connected"}
/health/readiness 200 172.20.0.5:4000 {"status":"healthy","db":"connected"}
```

Two different addresses answered. That is the load balancer at work.

### Part 2: the app only knows one URL

```python
import os
from openai import OpenAI

# The app only knows ONE address: the load balancer. It never sees the replicas.
client = OpenAI(
    base_url="http://localhost:4012",
    api_key=os.environ["LITELLM_MASTER_KEY"],
)

for number in range(4):
    raw = client.chat.completions.with_raw_response.create(
        model="fast-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    replica = raw.headers.get("X-Served-By")
    answer = raw.parse().choices[0].message.content
    print("request", number, "served by", replica, "->", answer)
```

Output (real, captured with litellm 1.104.0):

```text
request 0 served by 172.20.0.4:4000 -> Hello from the mock model
request 1 served by 172.20.0.5:4000 -> Hello from the mock model
request 2 served by 172.20.0.4:4000 -> Hello from the mock model
request 3 served by 172.20.0.5:4000 -> Hello from the mock model
```

With a real provider (not run here): drop `mock_response` and set `model: ollama/llama3.2` with `api_base: http://host.docker.internal:11434`, or a real provider model with `api_key: os.environ/OPENAI_API_KEY`. The app code does not change.

### Part 3: why replicas need shared Postgres and Redis

Create a virtual key with a limit of 4 requests per minute. Then use it 8 times. nginx alternates between the replicas.

```python
import os
import httpx

GATEWAY = "http://localhost:4012"
admin_headers = {"Authorization": "Bearer " + os.environ["LITELLM_MASTER_KEY"]}

# Part A: create a virtual key with a limit of 4 requests per minute.
created = httpx.post(
    GATEWAY + "/key/generate",
    headers=admin_headers,
    json={"rpm_limit": 4},
)
new_key = created.json()["key"]
print("key created by replica", created.headers.get("X-Served-By"))

# Part B: use that key 8 times. nginx alternates between the 2 replicas.
user_headers = {"Authorization": "Bearer " + new_key}
body = {"model": "fast-model", "messages": [{"role": "user", "content": "hi"}]}

for number in range(8):
    response = httpx.post(GATEWAY + "/chat/completions", headers=user_headers, json=body)
    replica = response.headers.get("X-Served-By")
    print("request", number, "replica", replica, "status", response.status_code)
```

Output with shared Redis (real, captured with litellm 1.104.0):

```text
key created by replica 172.20.0.4:4000
request 0 replica 172.20.0.5:4000 status 200
request 1 replica 172.20.0.4:4000 status 200
request 2 replica 172.20.0.5:4000 status 200
request 3 replica 172.20.0.4:4000 status 200
request 4 replica 172.20.0.5:4000 status 429
request 5 replica 172.20.0.4:4000 status 429
request 6 replica 172.20.0.5:4000 status 429
request 7 replica 172.20.0.4:4000 status 429
```

Read it slowly:

- The key was made on `.4`, and `.5` accepted it on the very next request. That is **shared Postgres**.
- Exactly 4 requests passed in total, spread over two replicas. That is **shared Redis**. The counter lives there:

```text
$ docker exec lm-ch12-redis redis-cli --scan --pattern '*requests*'
{api_key:be86a96fd842c0636079f7d6644e31b3451ab27a109dc980c822dcc6d97d7ea4}:requests
```

Now the same script with Redis removed from the config (`PROXY_CONFIG=./config-noredis.yaml docker compose up -d proxy1 proxy2`, a copy of `config.yaml` without `router_settings` and `coordination_redis`):

Output without Redis (real, captured with litellm 1.104.0):

```text
key created by replica 172.20.0.5:4000
request 0 replica 172.20.0.4:4000 status 200
request 1 replica 172.20.0.5:4000 status 200
request 2 replica 172.20.0.4:4000 status 200
request 3 replica 172.20.0.5:4000 status 200
request 4 replica 172.20.0.4:4000 status 200
request 5 replica 172.20.0.5:4000 status 200
request 6 replica 172.20.0.4:4000 status 200
request 7 replica 172.20.0.5:4000 status 200
```

A limit of 4 let 8 through. Each replica counted only its own 4. With 10 replicas, a "4 per minute" key would get 40. This is the single most important lesson of the chapter.

### Part 4: liveness vs readiness when Postgres dies

Liveness asks "is the process alive?". Readiness asks "should it get traffic?". They give different answers when a dependency fails.

```python
import httpx

GATEWAY = "http://localhost:4012"

live = httpx.get(GATEWAY + "/health/liveliness", timeout=30)
print("liveliness:", live.status_code, live.text)

ready = httpx.get(GATEWAY + "/health/readiness", timeout=30)
print("readiness: ", ready.status_code, ready.text)
```

Run once normally, then `docker stop lm-ch12-postgres`, wait 20 seconds, run again. The two `---` labels are mine, added between runs.

Output (real, captured with litellm 1.104.0):

```text
--- postgres up
liveliness: 200 "I'm alive!"
readiness:  200 {"status":"healthy","db":"connected"}
--- postgres stopped 20s
liveliness: 200 "I'm alive!"
readiness:  503 {"status":"healthy","db":"disconnected"}
```

Three details from this run:

- Liveness stayed 200. Kubernetes will not restart a pod just because the database is down. Restarting would not fix the database.
- Readiness returned **503**, so a load balancer stops sending traffic to that pod. Probes look at the **status code**, not the body (the body still says `"healthy"`).
- The first run right after stopping Postgres still said `connected`. The source caches a good DB result for 15 seconds (`_db_health_readiness_check_unbounded` in `_health_endpoints.py`). Probes react with a delay, not instantly.

If you set `general_settings.allow_requests_on_db_unavailable: true`, readiness stays 200 when the DB is down (checked in `PrismaDBExceptionHandler`). That trades safety for uptime: already-cached keys keep working.

```mermaid
flowchart TD
    K[Kubernetes kubelet] -->|every 15s| L[/health/liveliness/]
    K -->|every 10s| RD[/health/readiness/]
    L -->|fails 5 times| Restart[Restart the container]
    RD -->|503| Remove[Remove pod from Service endpoints]
    RD -->|200 again| Add[Send traffic again]
```

The periods (15s, 10s) and failure counts (5) come from the chart defaults shown in Part 7.

### Part 5: `num_workers` — processes inside one container

`--num_workers N` (env `NUM_WORKERS`, default 1, verified in `proxy_cli.py`) starts N server processes inside one container. I ran one container (mock model, no DB, no Redis) with `--num_workers 2` and kept only the start-up lines and the warnings:

Output (real, captured with litellm 1.104.0):

```text
INFO:     Started parent process [1]
INFO:     Started server process [11]
16:31:59 - LiteLLM Proxy:WARNING: login_throttle.py:112 - Running 2 workers but Redis is not configured. Failed Admin UI sign-in attempts are counted per worker, so the effective limits are 2 times the configured values. Configure Redis to share one count across workers.
INFO:     Started server process [12]
16:32:05 - LiteLLM Proxy:WARNING: login_throttle.py:112 - Running 2 workers but Redis is not configured. Failed Admin UI sign-in attempts are counted per worker, so the effective limits are 2 times the configured values. Configure Redis to share one count across workers.
```

Two lessons. First, workers are like replicas inside one box: they also have private memory, so the Redis rule from Part 3 applies to them too. The proxy itself warns you. Second, the production guide says: on Kubernetes, run **one worker per pod** and scale with more pods, because the HPA reads CPU against one process and rolling restarts stay smooth. On a single VM with nothing scaling it, set workers to the vCPU count ([docs.litellm.ai/docs/proxy/prod](https://docs.litellm.ai/docs/proxy/prod), "Workers and scaling").

### Part 6: load testing with Locust

**Locust** is a Python load-testing tool. You describe what one fake user does, then ask for hundreds of them. Install it in its own venv (`uv pip install locust`; I used locust 2.46.7).

```python
import os
from locust import HttpUser, task, between


class ChatUser(HttpUser):
    # Each fake user waits 0.1-0.5 s between requests, like a busy app.
    wait_time = between(0.1, 0.5)

    def on_start(self):
        api_key = os.environ["LITELLM_MASTER_KEY"]
        self.client.headers["Authorization"] = "Bearer " + api_key

    @task
    def chat(self):
        body = {
            "model": "fast-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        self.client.post("/chat/completions", json=body, name="chat")
```

Run without the web UI (`--headless`): 200 users, 20 new users per second, 60 seconds.

```bash
locust -f locustfile.py --headless -u 200 -r 20 -t 60s --host http://localhost:4012 --only-summary
```

Output, 2 replicas (real, captured with litellm 1.104.0):

```text
Type     Name    # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
POST     chat     21934     0(0.00%) |    203       2    2935    180 |  366.10        0.00

Type     Name      50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
POST     chat      180    220    230    250    340    500    730   1100   1900   2600   2900  21934
```

Then I switched nginx to send everything to one replica (`NGINX_CONF=./nginx-one.conf`, where `proxy2` is marked `down`) and ran the same test.

Output, 1 replica (real, captured with litellm 1.104.0):

```text
Type     Name    # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
POST     chat     12524     0(0.00%) |    578       2    4955    410 |  208.96        0.00

Type     Name      50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
POST     chat      410    530    640    740   1200   1600   2200   2800   4100   4900   5000  12524
```

(Column padding shortened; numbers unchanged.)

| | 1 replica | 2 replicas |
|---|---|---|
| Throughput | 209 req/s | 366 req/s |
| Median | 410 ms | 180 ms |
| p95 | 1600 ms | 500 ms |
| Failures | 0 | 0 |

How to read this honestly:

- These are **laptop numbers** (Docker Desktop, 10 CPUs shared by Locust, nginx, Postgres, Redis, and the proxies). They show the *shape*, not production capacity.
- The mock model answers instantly, so this measures **gateway overhead only**. With a real model, most of the time is the provider.
- The 2-replica run was not twice as fast. Something else (the laptop CPU, Postgres, Locust itself) starts to limit. Finding that "something else" is what load testing is for.
- A warm-up run with 50 users gave 156 req/s, median 7 ms. That run was limited by the users' wait time, not by the proxy. If your RPS equals `users ÷ average wait`, you have not found the limit yet.

Published numbers, for comparison: LiteLLM's benchmark page reports 2 instances on 4 CPU / 8 GB machines against a fake OpenAI endpoint at about 1,035 req/s, median 200 ms, with median LiteLLM overhead of 12 ms ([docs.litellm.ai/docs/benchmarks](https://docs.litellm.ai/docs/benchmarks)).

### Part 7: the Helm chart

The official chart is `litellm-helm`. Its source lives at [`helm/litellm-helm`](https://github.com/BerriAI/litellm/tree/main/helm/litellm-helm) in the LiteLLM repo, and it is published as an OCI package at `oci://ghcr.io/berriai/litellm-helm`, with chart versions that match LiteLLM releases ([docs.litellm.ai/docs/proxy/deploy](https://docs.litellm.ai/docs/proxy/deploy)). There is also a newer "componentized" chart that splits gateway, backend, and UI; this chapter uses the simpler monolithic one.

`helm` is not installed on this machine, so I ran it from the `alpine/helm` Docker image. `helm template` only renders YAML. It does not touch a cluster.

`my-values.yaml`:

```yaml
replicaCount: 3                       # three proxy pods

image:
  repository: ghcr.io/berriai/litellm
  tag: "v1.104.0"                     # pin the exact version

masterkeySecretName: litellm-masterkey   # Secret created with kubectl beforehand
masterkeySecretKey: masterkey

db:
  useExisting: true                   # we bring our own Postgres
  deployStandalone: false
  endpoint: "postgres.example.internal"
  database: litellm
  secret:
    name: litellm-db
    usernameKey: username
    passwordKey: password

environmentSecrets:
  - litellm-env                       # provider keys, REDIS_PASSWORD, salt key

proxy_config:
  model_list:
    - model_name: fast-model
      litellm_params:
        model: openai/gpt-4o-mini
        api_key: os.environ/OPENAI_API_KEY
  router_settings:
    redis_host: "redis.example.internal"
    redis_port: 6379
    redis_password: os.environ/REDIS_PASSWORD
  general_settings:
    master_key: os.environ/PROXY_MASTER_KEY

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

```bash
docker run --rm -v "$PWD":/work -w /work alpine/helm:latest \
  template ai-gw oci://ghcr.io/berriai/litellm-helm --version 1.104.0 -f my-values.yaml
```

Output (real, captured with litellm-helm chart 1.104.0; condensed to one line per object, then excerpts with some lines trimmed):

```text
# Source: litellm-helm/templates/configmap-litellm.yaml       kind: ConfigMap
# Source: litellm-helm/templates/service.yaml                 kind: Service
# Source: litellm-helm/templates/deployment.yaml              kind: Deployment
# Source: litellm-helm/templates/hpa.yaml                     kind: HorizontalPodAutoscaler
# Source: litellm-helm/templates/migrations-job.yaml          kind: Job
# Source: litellm-helm/templates/tests/test-connection.yaml   kind: Pod
# Source: litellm-helm/templates/tests/test-env-vars.yaml     kind: Pod
```

Deployment excerpt (not run on a cluster):

```yaml
          image: "ghcr.io/berriai/litellm:v1.104.0"
          env:
            - name: DATABASE_URL
              value: "postgresql://$(DATABASE_USERNAME):$(DATABASE_PASSWORD)@$(DATABASE_HOST)/$(DATABASE_NAME)"
            - name: PROXY_MASTER_KEY
              valueFrom:
                secretKeyRef:
                  name: litellm-masterkey
                  key: masterkey
            - name: DISABLE_SCHEMA_UPDATE
              value: "true"
          envFrom:
            - secretRef:
                name: litellm-env
          livenessProbe:
            httpGet:
              path: "/health/liveliness"
              port: "http"
            periodSeconds: 15
            failureThreshold: 5
          readinessProbe:
            httpGet:
              path: "/health/readiness"
              port: "http"
            periodSeconds: 10
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: "/health/readiness"
              port: "http"
            periodSeconds: 10
            failureThreshold: 30
```

HPA excerpt (not run on a cluster):

```yaml
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-gw-litellm
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

Map each object to the jargon table:

- **ConfigMap** `ai-gw-litellm-config` holds your `proxy_config` as `config.yaml`, mounted into every pod.
- **Secrets** are referenced, never inlined: `litellm-masterkey`, `litellm-db`, `litellm-env`. You create them with `kubectl create secret` before installing.
- **Deployment** runs the pods with all three probes. Notice it has **no `replicas:` line**: with autoscaling on, the HPA owns the count, and `replicaCount` is ignored.
- **Job** runs database migrations once. The pods get `DISABLE_SCHEMA_UPDATE=true` so 10 pods do not race to change the schema at the same moment.
- **Service** gives the stable name `ai-gw-litellm:4000`. An Ingress or cloud load balancer points at it.

## How big companies use this

- **Pfizer** runs LiteLLM as a self-hosted gateway "serving every team, application, and agent at Pfizer". Every LiteLLM version bump runs through Locust load tests in CI against a mock backend. That is how they caught a Redis TLS bug that dropped throughput from about 300 to 156 RPS **with zero HTTP errors**: Redis calls stalled, but every request still returned 200 ([LiteLLM blog, 2026-09-11](https://docs.litellm.ai/blog/pfizer-gateway-performance-and-resiliency)). Lesson: compare RPS against a baseline, not only the error rate.
- **Every Cure** (a team doing AI-driven drug repurposing) published its decision record: LiteLLM via the official Helm chart, managed by ArgoCD, `replicaCount: 3` for basic high availability, existing Postgres with PgBouncer, an in-cluster Redis for `router_settings.redis_host` and caching, and secrets synced from GCP Secret Manager ([Every Cure ADR](https://docs.dev.everycure.org/infrastructure/architecture_decision_records/deploying_LiteLLM/)).
- **Typical pattern** (from LiteLLM's own guides): one worker per pod, scale on CPU around 60%, do not scale on memory, and cap `maxReplicas` by what Postgres can serve. Each worker opens up to 10 DB connections by default, so 100 replicas could ask for about 1,000 connections ([prod guide](https://docs.litellm.ai/docs/proxy/prod)). Large deployments add PgBouncer and scale on requests per second instead of CPU ([deploy guide](https://docs.litellm.ai/docs/proxy/deploy), [benchmarks](https://docs.litellm.ai/docs/benchmarks)).
- **Typical pattern:** Prometheus scrapes `/metrics` from every pod, Grafana shows request rate, failures, and latency per model. LiteLLM ships ready Grafana dashboards in the repo under `cookbook/litellm_proxy_server/grafana_dashboard` ([Prometheus docs](https://docs.litellm.ai/docs/proxy/prometheus)).

## Traps

1. **Adding replicas without Redis.** Everything "works", but every limit is multiplied by the replica count (Part 3: limit 4, got 8). Nothing errors, so nobody notices until the bill arrives.
2. **Thinking `router_settings.redis_host` covers everything.** It shares Router state. Key and team limits and spend counters use the coordination Redis: `general_settings.coordination_redis`, a Redis `cache`, or, as a last resort, `REDIS_HOST`/`REDIS_PORT` env vars picked up silently. That last one surprised me in testing: a leftover `REDIS_HOST` env var made limits shared even though the config had no Redis. Be explicit.
3. **Using readiness as liveness.** If liveness checked the database, a Postgres blip would restart every pod at once. That turns a small outage into a big one. Liveness is "process alive"; readiness is "safe to receive traffic".
4. **Load testing only for errors.** A 0% error rate with half the RPS is still a regression (the Pfizer story). Also check that you are not limited by your own Locust settings: if RPS ≈ users ÷ wait time, add users.
5. **Many workers per pod with small resources.** Memory and CPU guidance is per worker (the prod guide says about 1 vCPU and 4 GiB each). Eight workers in a 1-CPU pod fight each other, and the HPA cannot read the situation well.
6. **Floating image tags.** `main-latest` changes under you. Pin `v1.104.0` in the values file so every pod runs the same code and rollbacks are exact.

## Project brief: L1

**Goal:** build a company AI platform: a LiteLLM gateway that runs as several replicas on a local Kubernetes cluster, shares state correctly, recovers from failure, has dashboards, and is proven by a load test.

**Features**

- A local cluster on your machine (Docker Desktop's built-in Kubernetes is enough; kind is fine if you already have it).
- LiteLLM installed with the official `litellm-helm` chart, version pinned, at least 2 replicas, values in your own `values.yaml`.
- Postgres and Redis running in the cluster (or as compose containers the cluster can reach). Both wired in: keys in Postgres, coordination and router state in Redis.
- At least two model names: one mock model for load tests, one real (Ollama or a free tier) for manual checks.
- Probes from the chart working; an HPA on CPU with a sane `maxReplicas`.
- Prometheus scraping the proxy and a Grafana dashboard showing request rate, failures, and latency.
- A Locust file plus a short `LOADTEST.md` with your numbers for 1 replica vs N replicas.

**Rules**

- No keys in code or in `values.yaml`. Provider keys and the master key go in Kubernetes Secrets created from a `.env` file that is git-ignored.
- Pin every image and chart version.
- Every number in `LOADTEST.md` must come from a run you did; write down the machine, users, spawn rate, and duration next to it.

**Done when**

- [ ] `kubectl get pods` shows 2+ LiteLLM pods `Running` and `READY 1/1`.
- [ ] A key created with `/key/generate` works on every pod (prove it by calling pods directly with `kubectl port-forward` to each pod).
- [ ] A key with `rpm_limit: 4` gets exactly 4 successes out of 8 calls spread over pods.
- [ ] `kubectl delete pod <one-litellm-pod>` during a Locust run: the run continues, and a new pod appears.
- [ ] Stopping Postgres makes `/health/readiness` return 503 while `/health/liveliness` stays 200.
- [ ] Grafana shows the Locust traffic as a rising request-rate line.
- [ ] `LOADTEST.md` has a 1-replica vs N-replica table with RPS, median, p95, and failures.
- [ ] `git grep -n "sk-"` in your repo finds no real keys.

**Hints**

1. Which Kubernetes object should hold `config.yaml`, and which should hold `OPENAI_API_KEY`? Why are they different?
2. When `autoscaling.enabled` is true, what happens to `replicaCount`? Run `helm template` and look.
3. How can you send a request to one specific pod instead of through the Service?
4. If your Locust RPS stops growing when you add replicas, what else on your machine could be the bottleneck?
5. What does `maxReplicas × workers × database_connection_pool_limit` tell you about your Postgres?

## Check yourself

1. Why does a key created through one replica work on another replica?

<details><summary>Answer</summary>
Keys are stored in Postgres, and every replica reads the same Postgres. The replicas themselves keep nothing important in memory (they are stateless), so any replica can check any key.
</details>

2. You run 5 replicas without Redis. A key has `rpm_limit: 10`. Roughly how many requests per minute can it make?

<details><summary>Answer</summary>
Up to about 50. Each replica counts only the requests it saw, so each allows 10. In Part 3, a limit of 4 let 8 through with 2 replicas.
</details>

3. Postgres goes down. What should the liveness and readiness probes report, and why?

<details><summary>Answer</summary>
Liveness: 200, because the process is fine and restarting it would not fix the database. Readiness: 503, so the load balancer stops sending traffic to that pod. This is exactly what LiteLLM 1.104.0 did in Part 4 (after its 15-second DB health cache expired).
</details>

4. On Kubernetes, should you run 1 pod with 8 workers or 8 pods with 1 worker? Why?

<details><summary>Answer</summary>
LiteLLM's production guide recommends 1 worker per pod and more pods. The HPA reads CPU against one process, resource sizing stays simple (about 1 vCPU and 4 GiB per worker), and rolling restarts drain one small pod at a time.
</details>

5. In the rendered Helm output, the Deployment has no `replicas:` field. Why?

<details><summary>Answer</summary>
Autoscaling was enabled, so the HorizontalPodAutoscaler owns the replica count (between `minReplicas: 3` and `maxReplicas: 10`). If the Deployment also set replicas, the two would fight on every upgrade.
</details>

6. Your load test shows 0% errors but RPS fell from 300 to 156 after an upgrade. Is that fine?

<details><summary>Answer</summary>
No. It is a serious regression: the same hardware now serves half the traffic, and latency rises under load. Pfizer caught exactly this with CI load tests; the cause was Redis calls stalling while every request still returned 200. Always compare RPS and latency against a saved baseline, not only the error rate.
</details>
