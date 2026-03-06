# Corporate SSL Certificates

## Purpose
This directory contains SSL certificates from your corporate PKI that need to be trusted by the dev container. These certificates are required for MCP servers to connect to internal services like `frodo.capgemini.com`.

## Required Certificates

The following certificates are needed for SSL verification to work:

| File | Description | Status |
|------|-------------|--------|
| `CapgeminiPKIRootCA.crt` | Root CA certificate | ✅ Included |
| `CapgeminiPKIIssuingCA1.crt` | Intermediate CA certificate | ✅ Included |

## How It Works

1. The Dockerfile copies all `.crt` files from `ssl-certs/` to `/usr/local/share/ca-certificates/`
2. During build, `update-ca-certificates` adds them to the system trust store
3. The `containerEnv` in `devcontainer.json` points apps to the certificate bundle

## How to Obtain Certificates (If Missing)

### Method 1: Download from AIA URLs (Recommended)

The certificates can be downloaded directly from Capgemini's PKI endpoints:

```bash
# Download Intermediate CA (CapgeminiPKIIssuingCA1)
curl -s "http://ocsp.capgemini.com/CertEnroll/WCRFRPAR01.corp.capgemini.com_CapgeminiPKIIssuingCA1(1).crt" \
  -o /tmp/issuing-ca.crt
openssl x509 -in /tmp/issuing-ca.crt -inform DER -out ssl-certs/CapgeminiPKIIssuingCA1.crt -outform PEM

# Download Root CA (CapgeminiPKIRootCA)
curl -s "http://ocsp.capgemini.com/CertEnroll/WRCFRPAR01_CapgeminiPKIRootCA.crt" \
  -o /tmp/root-ca.crt
openssl x509 -in /tmp/root-ca.crt -inform DER -out ssl-certs/CapgeminiPKIRootCA.crt -outform PEM
```

### Method 2: Extract from Windows

1. Open Chrome/Edge and visit `https://frodo.capgemini.com`
2. Click the lock icon → "Connection is secure" → "Certificate is valid"
3. Go to "Certification Path" tab
4. For each certificate in the chain (except the leaf):
   - Select the certificate → "View Certificate"
   - Go to "Details" tab → "Copy to File"
   - Export as "Base-64 encoded X.509 (.CER)"
5. Rename `.cer` to `.crt` and place in `ssl-certs/`

### Method 3: Extract using OpenSSL

```bash
# Connect to the server and extract certificate info
echo | openssl s_client -connect frodo.capgemini.com:443 -showcerts 2>/dev/null \
  | openssl x509 -noout -text | grep -A3 "Authority Information Access"

# Use the CA Issuers URLs shown to download the certificates
```

## Verifying Installation

After rebuilding the container, verify SSL works:

```bash
# Test with OpenSSL
echo | openssl s_client -connect frodo.capgemini.com:443 2>&1 | grep "Verify return code"
# Should show: Verify return code: 0 (ok)

# Test with Node.js
node -e "require('https').get('https://frodo.capgemini.com/gitea', r => console.log('Status:', r.statusCode))"
# Should show: Status: 200

# Test with curl
curl -I https://frodo.capgemini.com/gitea
# Should return HTTP headers without SSL errors
```

## Troubleshooting

### SSL still fails after rebuild
1. Ensure certificate files are in PEM format (start with `-----BEGIN CERTIFICATE-----`)
2. Check that files have `.crt` extension
3. Verify the Dockerfile COPY command runs before `update-ca-certificates`

### "unable to get issuer certificate" error
This means the intermediate or root CA is missing. Download both certificates using Method 1 above.

### Node.js ignores system certificates
Ensure `NODE_EXTRA_CA_CERTS` is set in `devcontainer.json`:
```json
"containerEnv": {
  "NODE_EXTRA_CA_CERTS": "/etc/ssl/certs/ca-certificates.crt"
}
```

## Security Note

**Do NOT** use `NODE_TLS_REJECT_UNAUTHORIZED=0` in production. This disables all SSL verification and makes connections vulnerable to man-in-the-middle attacks. Always use proper CA certificates instead.

---

## Adapting for Another Project

To reuse this devcontainer configuration in a different project, you need to update four project-specific values in `devcontainer.json`. Replace all instances of the project name (`goose` or `generic-RAG-demo`) with your new project folder name.

### Required Changes

1. **Container name** (line 9):
   ```json
   "runArgs": ["--name", "your-project-name-dev"]
   ```

2. **Workspace folder** (line 27):
   ```json
   "workspaceFolder": "/workspaces/your-project-name"
   ```

3. **OpenCode config path** (line 52):
   ```json
   "OPENCODE_CONFIG": "/workspaces/your-project-name/.opencode/opencode.json"
   ```

4. **Git safe directory** (line 61):
   ```json
   "postStartCommand": "git config --global --add safe.directory /workspaces/your-project-name && echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc || true"
   ```

### Example

For a project named `generic-RAG-demo`, the changes would be:

- Container: `agents-container` → `generic-RAG-demo-dev`
- Paths: `/workspaces/coding_agent_playground` → `/workspaces/generic-RAG-demo`

### Quick Find & Replace

Use your editor's find-and-replace to update all instances:

- Find: `goose` (or your current project name)
- Replace: `your-project-name`
- Files: `devcontainer.json` only

**Note:** The SSL certificate configuration, features, extensions, and port forwarding remain the same across projects.

---

## Connecting from External Terminals

You can connect to the running dev container from any terminal application (Ghostty, Windows Terminal, iTerm2, etc.) without using VS Code.

### Method: Docker Exec

Once the dev container is running (started via VS Code), connect from any terminal:

```bash
# Connect directly using the fixed container name
docker exec -it -w /workspaces/coding_agent_playground agents-container zsh
```

### Create a Shell Alias (Optional)

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, or PowerShell profile):

**Bash/Zsh:**
```bash
alias goose='docker exec -it -w /workspaces/coding_agent_playground agents-container zsh'
```

**PowerShell:**
```powershell
function goose { docker exec -it -w /workspaces/coding_agent_playground agents-container zsh }
```

Then simply run `goose` from any terminal to connect.

### Using Dev Containers CLI

Alternatively, use the official CLI:

```bash
# Install the CLI
npm install -g @devcontainers/cli

# Connect to the running container
devcontainer exec --workspace-folder /path/to/goose zsh
```
