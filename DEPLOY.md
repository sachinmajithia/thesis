# Deploying to a public AWS EC2 server

This app depends on heavy ML packages (`torch`, `transformers`,
`sentence-transformers`) and downloads an NLLB-200 translation model
(~2.4GB) plus IndicSBERT on first run, so it needs real RAM and disk - not
a free-tier/serverless setup. This guide runs it in Docker behind Caddy,
which gets you free automatic HTTPS with no manual certificate steps.

## 1. Launch the EC2 instance

- **AMI**: Ubuntu 22.04 LTS (or newer).
- **Instance type**: `t3.large` (8GB RAM / 2 vCPU) minimum. `t3.xlarge`
  (16GB) is more comfortable if you're also running `train_indicbert.py` or
  `build_parallel_corpus.py` on the same box. If you set `SKIP_NMT_MODEL=1`
  (Dictionary + EBMT only, no neural translation), `t3.medium` (4GB) can
  work, but leaves little headroom.
- **Storage**: at least 30GB gp3 EBS volume. Docker image layers +
  downloaded model weights + your corpus data add up quickly.
- **Security group**: allow inbound
  - TCP 22 (SSH) - restrict to your own IP, not `0.0.0.0/0`
  - TCP 80 (HTTP) - from anywhere (needed for the Let's Encrypt challenge)
  - TCP 443 (HTTPS) - from anywhere
  - Do **not** open port 5000 publicly - only Caddy should be internet-facing.
- **Elastic IP**: allocate one and associate it with the instance, so the
  public IP is stable across stops/reboots. Point your domain's DNS **A
  record** at this Elastic IP before continuing (DNS can take a few minutes
  to propagate - Caddy's certificate request will fail until it resolves).

## 2. Install Docker on the instance

SSH in, then:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker   # or log out/in so the group change takes effect
docker compose version   # sanity check - should print a version, no error
```

## 3. Get the code onto the server

```bash
git clone https://github.com/sachinmajithia/thesis.git
cd thesis
git checkout claude/plagiarism-history-module-5m0vcd   # or main/master once merged
```

## 4. Configure secrets

```bash
cp .env.example .env
nano .env   # or vim/your editor of choice
```

Fill in at minimum:
- `DOMAIN` - your actual domain name (must already point at this server's
  Elastic IP)
- `SECRET_KEY` - generate one: `python3 -c "import secrets; print(secrets.token_hex(32))"`
- `GOOGLE_API_KEY` / `GOOGLE_SEARCH_ENGINE_ID` - without these, internet
  plagiarism search falls back to fake simulated results (the app still
  runs, but that feature won't be real). Get them from
  [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
  and [Programmable Search Engine](https://programmablesearchengine.google.com/).

`.env` is gitignored - never commit it.

## 5. Build and start

```bash
docker compose up -d --build
```

First boot is slow: installing `torch`/`transformers` and downloading the
NLLB-200 + IndicSBERT models (several GB total) over the instance's network
connection. Watch progress with:

```bash
docker compose logs -f app
```

Wait for a line like `ENHANCED INTEGRATED SYSTEM READY!` before expecting
the site to respond. Then visit `https://your-domain.com` - Caddy requests
the HTTPS certificate automatically on first real request.

## 6. What persists across restarts

The whole project directory is bind-mounted into the `app` container (see
`docker-compose.yml`), so all of the following survive
`docker compose restart`, `docker compose down && up`, and instance
reboots, as long as the EBS volume itself isn't deleted:

- `corpus_database.db` (corpus documents + plagiarism check History)
- `uploads/`, `corpus/`, `data/*.csv`
- `models/` (a fine-tuned IndicBERT checkpoint, if you've trained one)

Downloaded model weights (NLLB-200, IndicSBERT) live in a separate named
Docker volume (`hf_cache`) so they survive rebuilds without cluttering the
project directory or `git status`.

**Back up the EBS volume periodically** (AWS Backup or manual snapshots) -
nothing here replicates data off the instance on its own.

## 7. Building the real 10,000+ pair corpus (optional)

This sandbox this project was developed in has no internet access to
huggingface.co, so `build_parallel_corpus.py` could only be written and
logic-tested there, never actually run against real data. Your EC2 instance
has normal internet access, so you can run it for real:

```bash
docker compose exec app python build_parallel_corpus.py --target_pairs 10000
docker compose restart app   # picks up data/parallel_corpus_extended.csv etc. on next startup
```

## 8. Deploying code updates

```bash
git pull
docker compose up -d --build
```

The bind-mounted data directories are untouched by this - only the code
and Python dependencies get rebuilt.

## 9. Cost notes

- Stop (don't just leave idle) the EC2 instance when you're not using it to
  avoid ongoing compute charges - an Elastic IP has a small hourly charge
  while *not* attached to a running instance, but no charge while attached.
- `t3.large` on-demand pricing is roughly $0.08/hr (region-dependent) -
  check current [AWS pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
  for your region.
- Google Custom Search's free tier is 100 queries/day; each plagiarism
  check can use several queries (exact-phrase + whole-paragraph +
  English-gloss, times up to two search passes) - watch usage if this gets
  real traffic.

## Troubleshooting

- **502 from Caddy right after `docker compose up`**: the `app` container is
  still loading models (`docker compose logs -f app` to watch). This can
  take a few minutes on first boot.
- **Caddy can't get a certificate**: confirm your domain's DNS A record
  actually resolves to this instance's Elastic IP (`dig your-domain.com`),
  and that ports 80/443 are open in the security group.
- **Container gets OOM-killed**: the instance doesn't have enough RAM for
  NLLB-200. Either upgrade the instance type, or set `SKIP_NMT_MODEL=1` in
  `.env` and `docker compose up -d --build` again (Dictionary + EBMT
  translation still work; NMT mode won't).
