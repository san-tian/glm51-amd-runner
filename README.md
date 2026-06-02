# glm51-amd-runner

GLM-5.1-FP8 on AMD runner for SGLang + ATOM PR355 OOT, with optional Tinker LoRA merge and FP8 quant.

## Version

- Current local recipe ref: `v2026-06-02-sglang-quark-loader-patch-v3.13`
- Entry point: `bootstrap.sh`

## Run

Pin `RECIPE_REF` to a tag or commit when running an experiment. Do not use a moving branch for recorded benchmark runs.

```bash
export CONTROL_PLANE_DIR="${CONTROL_PLANE_DIR:-/data/glm51-control}"
export RECIPE_REPO="${RECIPE_REPO:-https://github.com/san-tian/glm51-amd-runner}"
export RECIPE_REF="${RECIPE_REF:-v2026-06-02-sglang-quark-loader-patch-v3.13}"
export RECIPE_ARCHIVE_URL="${RECIPE_ARCHIVE_URL:-${RECIPE_REPO%.git}/archive/${RECIPE_REF}.tar.gz}"

sudo install -d -m 0755 "$CONTROL_PLANE_DIR"
cd "$CONTROL_PLANE_DIR"

curl -fsSLo recipe.tar.gz "$RECIPE_ARCHIVE_URL"
sudo rm -rf recipe
mkdir -p recipe
tar -xzf recipe.tar.gz -C recipe --strip-components=1

bash recipe/bootstrap.sh
```

## Secrets

Do not commit secrets. Put runtime secrets on the AMD server:

```bash
export CONTROL_PLANE_DIR="${CONTROL_PLANE_DIR:-/data/glm51-control}"
sudo install -d -m 0700 "${CONTROL_PLANE_DIR}/secrets"
sudo tee "${CONTROL_PLANE_DIR}/secrets/glm51-secrets.env" >/dev/null <<'EOF'
HF_TOKEN=<hf token>
TINKER_URL=<tinker://.../weights/...>
RUN_TINKER_MERGE_QUANT=auto
EOF
sudo chmod 0600 "${CONTROL_PLANE_DIR}/secrets/glm51-secrets.env"
```

## Validate

```bash
bash scripts/check.sh
```
