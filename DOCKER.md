# Docker setup guide for Morphik Core

This guide uses the production Compose deployment in `docker-compose.run.yml`. It runs the published Morphik Core
image with PostgreSQL and Redis. Ollama and the admin UI are optional profiles.

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 10GB of free disk space (for models and data)
- 8GB+ RAM recommended

## Quick start

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/morphik-org/morphik-core.git
cd morphik-core
```

2. Run the production installer:

```bash
./install_docker.sh
```

The installer pulls the published image, creates `.env`, writes the Docker configuration, asks which model provider to
use, and starts `docker-compose.run.yml`.

3. For subsequent runs:

```bash
./start-morphik.sh
./stop-morphik.sh
```

Both commands are idempotent. The stop script removes containers and the Compose network, including services in
optional profiles, but preserves the named volumes that hold PostgreSQL, Redis, and model data.

4. To completely reset all Compose-managed data, run this destructive command:

```bash
docker compose -f docker-compose.run.yml --profile "*" down --volumes --remove-orphans
```

It removes PostgreSQL and every other named volume. Run `./morphik-backup.sh backup` before an intentional reset.
Do not add `--volumes` to a normal shutdown. The generated `stop-morphik` script preserves all data volumes.

Installers before September 4, 2026 (PR #435) generated a `stop-morphik.sh` that ran `down --volumes`, so every stop
deleted the database. Check an older install with `grep -n -- --volumes stop-morphik.sh`. If it matches, replace
the script with the current one before you stop Morphik again.

## Configuration

### 1. Default setup

The installed stack includes:

- PostgreSQL with pgvector for document storage
- Redis for ingestion jobs
- Local file storage
- Configurable local or external model providers
- Token authentication or explicit local-only bypass mode

### 2. Configuration file

Edit the generated `morphik.toml`. For example, an Ollama container on the same Compose network uses:

```toml
[api]
host = "0.0.0.0"
port = 8000

[registered_models]
ollama_chat = { model_name = "ollama_chat/llama3.2", api_base = "http://ollama:11434" }
ollama_embedding = { model_name = "ollama/nomic-embed-text", api_base = "http://ollama:11434" }

[completion]
model = "ollama_chat"

[embedding]
model = "ollama_embedding"
dimensions = 768
similarity_metric = "cosine"

[database]
provider = "postgres"

[vector_store]
provider = "pgvector"

[storage]
provider = "local"
storage_path = "/app/storage"

[morphik]
mode = "self_hosted"
enable_colpali = false
colpali_mode = "off"
```

Set `COMPOSE_PROFILES=ollama` in `.env` before starting this example. Pull the required models into Ollama before using
the API.

### 3. Environment variables

The installer creates `.env`. Review these values before exposing the service:

```bash
JWT_SECRET_KEY=your-secure-key-here  # Important: Change in production
OPENAI_API_KEY=sk-...                # Only if using OpenAI
TELEMETRY=false                      # Recommended for a no-egress deployment
COMPOSE_PROJECT_NAME=morphik         # Set once before the first start
```

### 4. Custom configuration

`docker-compose.run.yml` mounts `./morphik.toml` read-only into both the API and worker containers. Edit that file and
run `./start-morphik.sh` again to apply changes.

## Accessing services

- Morphik API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Storage and data

- Database data: Stored on the host in the `postgres_data` Docker volume. It survives container replacement and `docker compose down`.
- AI Models: Stored in the `ollama_data` Docker volume
- Documents: Stored in `./storage` directory (mounted to container)
- Logs: Available in `./logs` directory

To keep the PostgreSQL files in a visible host directory on a new installation, set this in `.env` before the first start:

```bash
MORPHIK_POSTGRES_DATA_PATH=./postgres-data
```

Do not add or change this setting on an existing installation until you have migrated the current database. Pointing Postgres at an empty directory creates an empty database and makes the existing data appear lost.

The generated start and stop scripts also recover the existing Compose project name from Docker's container and volume labels. This prevents a moved installation directory from leaving the old containers and database volume behind.

## Backup and restore

`morphik-backup.sh` backs up and restores the whole knowledge base. The installer downloads it next to
`start-morphik.sh`. It needs only bash and Docker on the host. Run it from the install directory.

### Back up

```bash
./morphik-backup.sh backup
```

This writes one file, such as `backups/morphik-20260924T120000Z.backup`. The file is a tar archive with:

- `database.dump`: a `pg_dump -Fc` of PostgreSQL. It has documents, metadata, folders, chunks, and embeddings.
  ColPali embeddings are included when `multivector_store` is `postgres`.
- `storage.tar`: the `./storage` directory with the uploaded source files.
- `config/morphik.toml`.
- `manifest.json`: the Core image, embedding model and dimensions, store providers, document counts by status,
  row counts and checksums for every table, and a sha256 for each part.

The script dumps PostgreSQL first and copies storage second, so every file the dump references is in the archive.
The dump and the manifest counts come from one database snapshot. You can back up while Morphik is running.

`.env` is left out because it holds secrets. Add `--include-env` to keep it, and store that backup like a password.
Backup files and the `backups` directory are readable only by their owner.

Redis is not backed up. It holds only the ingestion queue.

Backups work the same when `MORPHIK_POSTGRES_DATA_PATH` keeps PostgreSQL in a host directory. The script reads
the database through PostgreSQL, not from its files.

If `[storage] provider` is `aws-s3`, the source files stay in the bucket and are not copied. The manifest records
this, and the script prints a warning. Turn on bucket versioning to protect those files.

A backup of an empty database prints a warning. That usually means `COMPOSE_PROJECT_NAME` or the install
directory name changed, and Compose is using a new, empty volume. See `docker volume ls` for the old one.

### Check a backup

```bash
./morphik-backup.sh verify backups/morphik-20260924T120000Z.backup
```

`verify` checks every sha256, restores the dump into a temporary PostgreSQL server inside a throwaway container,
and compares row counts and table checksums with the manifest. It also checks that each source file the database
references is in the archive. It does not touch the running deployment.

### Restore

```bash
./morphik-backup.sh restore backups/morphik-20260924T120000Z.backup
```

Restore validates the file first. It stops without changing anything when:

- a checksum does not match
- the embedding model or dimensions differ from `morphik.toml`. Restored embeddings would not match new queries.
- the vector store, multivector store, or storage provider differs from `morphik.toml`
- the deployment already has data in PostgreSQL or `./storage`

Add `--force` to replace existing data. Restore then writes a safety backup of the current data first, named
`*-pre-restore.backup`. `--no-safety-backup` skips it.

Once the checks pass, restore stops the `morphik` and `worker` services. It recreates the database, runs
`pg_restore --clean --if-exists --no-owner`, and compares the result with the manifest. Then it replaces
`./storage` and runs `./start-morphik.sh`. Nothing is re-embedded. Add `--no-start` to leave the services stopped.

Documents that were still ingesting when the backup was taken come back as `failed`, with their ingestion revision
increased by one. Any old queued job for them is then skipped. Restore writes their IDs to
`backups/restore-<time>-requeue.json` and prints the `POST /ingest/requeue` command that finishes them.

To rebuild on a new host, run the installer and then restore with `--restore-config`. That uses the
`morphik.toml` from the backup, and `.env` too if the backup has it. The replaced files are kept as
`*.before-restore-<time>`.

`./morphik-backup.sh list` shows the backups in the backup directory.

### Scheduled backups

Set this in `morphik.toml`, then run `./start-morphik.sh`:

```toml
[backup]
enabled = true
interval_hours = 12
directory = "./backups"
keep = 14
verify = false
s3_uri = ""
```

This starts the optional `backup` service. It runs the same `morphik-backup.sh` in the PostgreSQL image, outside
the API process, so a stuck API does not stop backups. The service reads `[backup]` before each run.

- `keep` applies only to scheduled backups, named `*-auto.backup`. Manual and pre-restore backups are never deleted.
- `verify = true` checks each new backup in a temporary database. It needs free disk space about the size of the
  database.
- `include_env = true` adds `.env` to scheduled backups.

Check it with `docker compose -f docker-compose.run.yml logs backup`.

### Off-host copies

A backup on the same disk is lost with the machine. Set an S3 destination:

```toml
[backup]
s3_uri = "s3://my-bucket/morphik"
s3_region = "us-east-1"
```

`./start-morphik.sh` then also starts the `backup-s3` service. It uploads new backups every 5 minutes.
`./morphik-backup.sh backup` uploads its file right away; `--no-upload` skips that. Uploads never delete remote
files. Use an S3 lifecycle rule for remote retention, and consider S3 Object Lock so a compromised host cannot
delete old backups.

Credentials come from the shell, then `.env`, then the EC2 instance role. The role needs `s3:PutObject` and
`s3:ListBucket` on the bucket. Containers reach the instance role only when the metadata hop limit is at least 2:

```bash
aws ec2 modify-instance-metadata-options --instance-id <instance-id> \
  --http-put-response-hop-limit 2 --http-endpoint enabled
```

## Troubleshooting

1. **Service will not start**

   ```bash
   # View all logs
   docker compose -f docker-compose.run.yml --profile "*" logs

   # View specific service logs
   docker compose -f docker-compose.run.yml logs morphik
   docker compose -f docker-compose.run.yml logs postgres
   docker compose -f docker-compose.run.yml --profile ollama logs ollama
   ```

2. **Database issues**

   - Check PostgreSQL health: `docker compose -f docker-compose.run.yml ps`
   - Verify the database: `docker compose -f docker-compose.run.yml exec postgres psql -U morphik -d morphik`

3. **Container name is already in use**

   - Do not delete the existing containers or volumes before identifying their Compose project.
   - Read the current project name: `docker inspect morphik-postgres --format '{{ index .Config.Labels "com.docker.compose.project" }}'`
   - List existing Postgres volumes: `docker volume ls --filter label=com.docker.compose.volume=postgres_data`
   - The generated scripts select the existing project automatically. If more than one project owns a Postgres volume, set the intended project explicitly in `.env` with `COMPOSE_PROJECT_NAME=<project>`.

4. **Model download issues**

   - Check Ollama logs: `docker compose -f docker-compose.run.yml --profile ollama logs ollama`
   - Ensure enough disk space for models
   - Restart Ollama: `docker compose -f docker-compose.run.yml --profile ollama restart ollama`

5. **Performance issues**

   - Monitor resources: `docker stats`
   - Ensure sufficient RAM (8GB+ recommended)
   - Check disk space: `df -h`

## Production deployment

For production environments:

1. **Security**

   - Change the default `JWT_SECRET_KEY`
   - Use proper network security groups
   - Enable HTTPS (recommended: use a reverse proxy)
   - Regularly update containers and dependencies

2. **Persistence**

   - Use named volumes for all data
   - Turn on scheduled backups with an off-host copy (see [Backup and restore](#backup-and-restore))
   - Test a restore with `./morphik-backup.sh verify` before upgrading production deployments

3. **Monitoring**

   - Set up container monitoring
   - Configure proper logging
   - Use health checks

## Support

For issues and feature requests:

- GitHub Issues: [https://github.com/morphik-org/morphik-core/issues](https://github.com/morphik-org/morphik-core/issues)
- Documentation: [https://docs.morphik.ai](https://docs.morphik.ai)

## Repository information

- License: MIT
