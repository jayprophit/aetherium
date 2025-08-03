---
title: Cloud Service Models and RDBMS
created_at: 2025-07-08
author: Knowledge Base System
description: Overview of IaaS, PaaS, SaaS, and RDBMS concepts for modern system design
version: 1.0.0
tags:
  - cloud
  - iaas
  - paas
  - saas
  - rdbms
  - databases
  - system_design
relationships:
  related:
    - database_overview.md
    - ../databases/database_overview.md
    - ../system_design/cache.md
    - ../system_design/denormalization.md
---

# Cloud Service Models: IaaS, PaaS, SaaS

Cloud computing offers three main service models, each providing different levels of abstraction, control, and management:

## 1. IaaS (Infrastructure as a Service)
- **What it provides:** Virtualized computing resources (servers, storage, networking)
- **User responsibility:** OS, middleware, applications, data
- **Provider responsibility:** Physical hardware, virtualization, networking
- **Use case:** Scalable infrastructure without managing physical servers
- **Examples:** AWS EC2, Azure Virtual Machines, Google Compute Engine

## 2. PaaS (Platform as a Service)
- **What it provides:** Platform for developing, testing, and deploying applications (runtime, tools, OS)
- **User responsibility:** Applications and data
- **Provider responsibility:** Servers, storage, networking, OS, middleware
- **Use case:** Developers focus on code, not infrastructure
- **Examples:** Heroku, Google App Engine, Azure App Service

## 3. SaaS (Software as a Service)
- **What it provides:** Ready-to-use software applications delivered over the internet
- **User responsibility:** Data and user configuration
- **Provider responsibility:** Everything (infrastructure, OS, software, updates)
- **Use case:** End-users or businesses needing accessible, hassle-free software
- **Examples:** Gmail, Salesforce, Microsoft 365, Slack

### Key Differences
| Model | Control Level | User Manages | Provider Manages |
|-------|--------------|--------------|------------------|
| IaaS  | High         | Apps, Data, OS | Hardware, Networking |
| PaaS  | Medium       | Apps, Data     | OS, Runtime, Tools   |
| SaaS  | Low          | Data           | Everything Else      |

#### Analogy (Pizza as a Service)
- **IaaS:** You get the raw ingredients (dough, sauce) – cook it yourself.
- **PaaS:** A half-baked pizza – you add toppings and bake.
- **SaaS:** A delivered pizza – just eat it!

Each model suits different needs, from full control (IaaS) to zero maintenance (SaaS).

---

# RDBMS (Relational Database Management System)

A system designed to store, manage, and query structured data using a relational model (tables with rows and columns).

## Key Features
- **Tables (Relations):** Data stored in rows (tuples) and columns (attributes)
- **Primary Keys:** Unique identifiers for each row
- **Foreign Keys:** Columns linking tables
- **SQL (Structured Query Language):** Standard language to interact with RDBMS
- **ACID Compliance:**
  - Atomicity (all-or-nothing execution)
  - Consistency (valid state after transactions)
  - Isolation (concurrent transactions don’t interfere)
  - Durability (committed data persists)

## Popular RDBMS Examples
| Name        | Developer   | Key Use Cases                  |
|-------------|-------------|-------------------------------|
| MySQL       | Oracle      | Web apps (WordPress, Shopify) |
| PostgreSQL  | Open-Source | Complex queries, geospatial   |
| SQL Server  | Microsoft   | Enterprise, Windows apps      |
| Oracle DB   | Oracle      | Large-scale enterprise        |
| SQLite      | Open-Source | Embedded/mobile apps          |

## RDBMS vs. NoSQL
| Feature      | RDBMS (SQL)         | NoSQL                  |
|--------------|---------------------|------------------------|
| Data Model   | Structured (tables) | Flexible (docs, graph) |
| Schema       | Fixed               | Dynamic (schema-less)  |
| Scalability  | Vertical            | Horizontal             |
| Use Cases    | Complex queries     | Big data, real-time    |
| Examples     | MySQL, PostgreSQL   | MongoDB, Cassandra     |

## Why RDBMS?
- Ideal for transactional systems (banking, e-commerce)
- Ensures data integrity via ACID
- Powerful querying with SQL

### Limitations
- Less flexible for unstructured data
- Harder to scale horizontally compared to NoSQL

#### Analogy
An RDBMS is like a filing cabinet with labeled folders (tables) and strict rules. NoSQL is like a storage room where you toss items in any box and reorganize later.

---

## See Also
- [Database Overview](../databases/database_overview.md)
- [Denormalization](denormalization.md)
- [Caching](cache.md)
- [Blob Storage](blob_storage.md)
