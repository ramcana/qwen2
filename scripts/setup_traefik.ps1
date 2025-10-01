# Traefik Setup Script for Windows PowerShell
# This script initializes the required directories and files for Traefik configuration

param(
    [switch]$Production,
    [string]$Domain = "localhost",
    [string]$Email = "admin@localhost"
)

Write-Host "🚀 Setting up Traefik configuration..." -ForegroundColor Green

# Create required directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
$directories = @(
    "config\traefik",
    "ssl\certs",
    "ssl\private", 
    "logs\traefik",
    "data\traefik"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✅ Created $dir" -ForegroundColor Green
    }
    else {
        Write-Host "  ℹ️  Directory $dir already exists" -ForegroundColor Cyan
    }
}

# Create ACME storage files
Write-Host "🔐 Creating ACME storage files..." -ForegroundColor Yellow
$acmeFiles = @("ssl\acme.json", "ssl\acme-dev.json")

foreach ($file in $acmeFiles) {
    if (!(Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "  ✅ Created $file" -ForegroundColor Green
    }
    else {
        Write-Host "  ℹ️  File $file already exists" -ForegroundColor Cyan
    }
}

# Generate self-signed certificates for development
Write-Host "🔑 Generating self-signed certificates for development..." -ForegroundColor Yellow

if (!(Test-Path "ssl\certs\localhost.crt")) {
    try {
        # Check if OpenSSL is available
        $opensslPath = Get-Command openssl -ErrorAction SilentlyContinue
        
        if ($opensslPath) {
            $subjectAltNames = "DNS:localhost,DNS:*.localhost,DNS:app.localhost,DNS:api.localhost,DNS:traefik.localhost,IP:127.0.0.1"
            
            & openssl req -x509 -nodes -days 365 -newkey rsa:2048 `
                -keyout "ssl\private\localhost.key" `
                -out "ssl\certs\localhost.crt" `
                -subj "/C=US/ST=Local/L=Local/O=Development/CN=localhost" `
                -addext "subjectAltName=$subjectAltNames"
            
            Write-Host "  ✅ Self-signed certificate generated with OpenSSL" -ForegroundColor Green
        }
        else {
            # Fallback to PowerShell certificate creation
            Write-Host "  ⚠️  OpenSSL not found, using PowerShell certificate creation" -ForegroundColor Yellow
            
            $cert = New-SelfSignedCertificate `
                -DnsName @("localhost", "*.localhost", "app.localhost", "api.localhost", "traefik.localhost") `
                -CertStoreLocation "cert:\LocalMachine\My" `
                -NotAfter (Get-Date).AddYears(1) `
                -KeyAlgorithm RSA `
                -KeyLength 2048 `
                -HashAlgorithm SHA256
            
            # Export certificate and private key
            $certPath = "ssl\certs\localhost.crt"
            $keyPath = "ssl\private\localhost.key"
            
            # Export certificate (public key)
            Export-Certificate -Cert $cert -FilePath $certPath -Type CERT | Out-Null
            
            # Export private key (requires password, using empty password for development)
            $password = ConvertTo-SecureString -String "" -Force -AsPlainText
            Export-PfxCertificate -Cert $cert -FilePath "ssl\private\localhost.pfx" -Password $password | Out-Null
            
            Write-Host "  ✅ Self-signed certificate generated with PowerShell" -ForegroundColor Green
            Write-Host "  ℹ️  Note: Private key exported as PFX format" -ForegroundColor Cyan
        }
    }
    catch {
        Write-Host "  ❌ Failed to generate certificate: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  ℹ️  You may need to install OpenSSL or run as Administrator" -ForegroundColor Cyan
    }
}
else {
    Write-Host "  ℹ️  Self-signed certificate already exists" -ForegroundColor Cyan
}

# Create Docker networks
Write-Host "🌐 Creating Docker networks..." -ForegroundColor Yellow
$networks = @("qwen-network", "traefik-public")

foreach ($network in $networks) {
    try {
        $existingNetwork = docker network ls --filter "name=$network" --format "{{.Name}}" 2>$null
        if ($existingNetwork -eq $network) {
            Write-Host "  ℹ️  Network $network already exists" -ForegroundColor Cyan
        }
        else {
            docker network create $network 2>$null
            Write-Host "  ✅ Created network $network" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "  ⚠️  Could not create network $network (Docker may not be running)" -ForegroundColor Yellow
    }
}

# Create .env file template
if (!(Test-Path ".env")) {
    Write-Host "📝 Creating .env template..." -ForegroundColor Yellow
    
    $envContent = @"
# Traefik Configuration
ACME_EMAIL=$Email
TRAEFIK_DOMAIN=traefik.$Domain
APP_DOMAIN=app.$Domain
API_DOMAIN=api.$Domain

# Production Configuration (uncomment for production)
# ACME_EMAIL=your-email@example.com
# TRAEFIK_DOMAIN=traefik.yourdomain.com
# APP_DOMAIN=app.yourdomain.com
# API_DOMAIN=api.yourdomain.com
# DNS_PROVIDER=cloudflare
# TRAEFIK_AUTH_USER=admin:`$2y`$10`$2b2cu/0P6dvFYOTwMZQo4OEbb.Npsb.2bJ65v2Oy7SHq6UfXVo0n2

# Optional: Jaeger tracing
# JAEGER_AGENT=jaeger:6831
"@
    
    Set-Content -Path ".env" -Value $envContent -Encoding UTF8
    Write-Host "  ✅ .env template created" -ForegroundColor Green
}
else {
    Write-Host "  ℹ️  .env file already exists" -ForegroundColor Cyan
}

# Create basic auth file for dashboard
Write-Host "🔒 Creating basic auth for dashboard..." -ForegroundColor Yellow
try {
    # Try to use htpasswd if available (from Apache utils or Git Bash)
    $htpasswdPath = Get-Command htpasswd -ErrorAction SilentlyContinue
    
    if ($htpasswdPath) {
        & htpasswd -cb "config\traefik\.htpasswd" admin admin 2>$null
        Write-Host "  ✅ Basic auth created with htpasswd" -ForegroundColor Green
    }
    else {
        # Fallback: create basic auth manually (bcrypt hash for 'admin')
        $authContent = "admin:`$2y`$10`$2b2cu/0P6dvFYOTwMZQo4OEbb.Npsb.2bJ65v2Oy7SHq6UfXVo0n2"
        Set-Content -Path "config\traefik\.htpasswd" -Value $authContent -Encoding UTF8
        Write-Host "  ✅ Basic auth created manually" -ForegroundColor Green
    }
}
catch {
    Write-Host "  ⚠️  Could not create basic auth file: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Traefik setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Review and update .env file with your domain settings"
Write-Host "2. For production, update ACME_EMAIL and domain settings"
Write-Host "3. Start Traefik with: docker-compose -f docker-compose.traefik.yml up -d"
Write-Host "4. Access dashboard at: https://traefik.$Domain`:8080 (admin/admin)"
Write-Host ""
Write-Host "🔧 Available commands:" -ForegroundColor Cyan
Write-Host "  Development: docker-compose -f docker-compose.traefik.yml up -d"
Write-Host "  Production:  docker-compose -f docker-compose.traefik.prod.yml up -d"
Write-Host "  Logs:        docker-compose -f docker-compose.traefik.yml logs -f traefik"
Write-Host "  Stop:        docker-compose -f docker-compose.traefik.yml down"

if ($Production) {
    Write-Host ""
    Write-Host "🏭 Production mode selected!" -ForegroundColor Magenta
    Write-Host "Remember to:" -ForegroundColor Yellow
    Write-Host "- Update domain settings in .env"
    Write-Host "- Configure DNS records"
    Write-Host "- Set up proper authentication"
    Write-Host "- Review security settings"
}