---
title: Load Balancer
date: 2025-07-08
---

# Load Balancer

---
id: web-system-design-load-balancer
title: Load Balancer - System Design
description: Comprehensive documentation on load balancers, types, algorithms, and
  implementation examples
author: Knowledge Base System
created_at: 2025-07-02
updated_at: 2025-07-02
version: 1.0.0
tags:
- system_design
- load_balancing
- high_availability
- scalability
- infrastructure
relationships:
  prerequisites: []
  successors: []
  related:
  - ../networking/proxy.md
  - ../scalability/horizontal_scaling.md
---

# Load Balancers

## Overview

Load balancers are critical components in distributed systems that distribute incoming network traffic across multiple servers to ensure no single server bears too much demand. They help maintain application availability, reliability, and scalability by preventing server overload and providing redundancy.

## Types of Load Balancers

### 1. Hardware Load Balancers
Physical devices optimized for high-performance traffic routing.

**Characteristics:**
- Dedicated hardware with specialized processors
- High throughput capacity
- Lower latency
- Typically more expensive

**Examples:**
- F5 BIG-IP
- Citrix ADC (formerly NetScaler)
- A10 Networks

### 2. Software Load Balancers
Software implementations that can be deployed on standard servers or as virtual appliances.

**Characteristics:**
- Flexible deployment options
- Cost-effective
- Easier to scale horizontally
- Configuration through API/software

**Examples:**
- NGINX
- HAProxy
- AWS Elastic Load Balancing
- Envoy

### 3. Layer 4 Load Balancers (Transport Layer)
Operate at the transport layer, distributing traffic based on network information like IP addresses and ports.

**Characteristics:**
- Protocol-agnostic (works with any TCP/UDP application)
- Simple and fast
- Limited content-based routing capabilities

### 4. Layer 7 Load Balancers (Application Layer)
Operate at the application layer, distributing requests based on application-specific content.

**Characteristics:**
- Content-based routing (URLs, HTTP headers, cookies)
- SSL termination capabilities
- More sophisticated request handling
- Higher processing overhead

## Load Balancing Algorithms

### 1. Round Robin
Requests are distributed sequentially across the server group.

```python
class RoundRobinLoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current_index = 0

    def get_next_server(self):
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
``````python
class LeastConnectionsLoadBalancer:
    def __init__(self, servers):
        self.servers = {server: 0 for server in servers}  # server: connection_count

    def get_next_server(self):
        server = min(self.servers.items(), key=lambda x: x[1])[0]
        self.servers[server] += 1
        return server

    def release_connection(self, server):
        self.servers[server] -= 1
``````python
class WeightedRoundRobinLoadBalancer:
    def __init__(self, server_weights):  # {server: weight}
        self.server_weights = server_weights
        self.servers = []
        for server, weight in server_weights.items():
            self.servers.extend([server] * weight)
        self.current_index = 0

    def get_next_server(self):
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
``````python
class IPHashLoadBalancer:
    def __init__(self, servers):
        self.servers = servers;
        
    def get_server_for_ip(self, ip_address):
        # Simple hash function
        hash_value = sum(int(octet) for octet in ip_address.split('.'));
        server_index = hash_value % len(self.servers);
        return self.servers[server_index]:
``````python
http {
    upstream backend {
        # Load balancing method
        least_conn;
        
        # List of backend servers
        server backend1.example.com weight=3;
        server backend2.example.com;
        server backend3.example.com;
        
        # Health checks
        server backend4.example.com max_fails=3 fail_timeout=30s;
    }
    
    server {
        listen 80;
        
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
``````python
version: '3'

services:
  haproxy:
    image: haproxy:latest
    ports:
      - "80:80"
      - "443:443"
      - "9000:9000"  # HAProxy stats
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg
    depends_on:
      - web1
      - web2
      - web3
  
  web1:
    image: nginx:alpine
    volumes:
      - ./web1:/usr/share/nginx/html
  
  web2:
    image: nginx:alpine
    volumes:
      - ./web2:/usr/share/nginx/html
  
  web3:
    image: nginx:alpine
    volumes:
      - ./web3:/usr/share/nginx/html
``````python
# NOTE: The following code had syntax errors and was commented out
# # NOTE: The following code had syntax errors and was commented out
# # global
# #     log /dev/log local0
# #     log /dev/log local1 notice
# #     daemon
# #     maxconn 4096
# # 
# # defaults
# #     log global
# #     mode http
# #     option httplog
# #     option dontlognull
# #     timeout connect 5000
# #     timeout client 50000
# #     timeout server 50000
# # 
# # frontend http_front
# #     bind *:80
# #     stats uri /haproxy?stats
# #     default_backend http_back
# # 
# # backend http_back
# #     balance roundrobin
# #     server web1 web1:80 check
# #     server web2 web2:80 check
# #     server web3 web3:80 check
# # 
# # listen stats
# #     bind *:9000
# #     stats enable
# #     stats uri /
# #     stats refresh 5s
# #     stats realm Haproxy\ Statistics
# #     stats auth admin:admin
``````python
import requests
import time
from threading import Thread

class HealthChecker(Thread):
    def __init__(self, servers, check_interval=5, timeout=2):
        super().__init__()
        self.servers = servers
        self.healthy_servers = {server: True for server in servers}
        self.check_interval = check_interval
        self.timeout = timeout
        self.daemon = True
        :
    def run(self):
        while True:
            for server in self.servers:
                try:
                    response = requests.get(f"http://{server}/health", timeout=self.timeout)
                    if response.status_code == 200:
                        self.healthy_servers[server] = True
                    else:
                        self.healthy_servers[server] = False
                except requests.RequestException:
                    self.healthy_servers[server] = False
            
            time.sleep(self.check_interval)
    
    def get_healthy_servers(self):
        return [server for server, healthy in self.healthy_servers.items() if healthy]:
```