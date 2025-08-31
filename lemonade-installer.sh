#!/bin/bash
# Lemonade SDK installer for Windows (via WSL)
# This script installs Lemonade on the Windows host when running from WSL

set -e

# Color functions
print_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
}

# Main installation function
install_lemonade() {
    print_info "🍋 Setting up Lemonade SDK for local inference..."

    # Check if we're in WSL
    if ! grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null && [ ! -f /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        print_error "This installer only works in WSL. For native Windows, install directly with pip."
        print_info "Installation command: pip install lemonade-sdk[oga-ryzenai] --extra-index-url=https://pypi.amd.com/simple"
        return 1
    fi

    print_info "Checking for Windows Python installation..."

    # Try to find Python on Windows via PowerShell
    WIN_PYTHON=$(powershell.exe -Command "& {
        \$pythonPath = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
        if (\$pythonPath) {
            Write-Output \$pythonPath
        } else {
            # Check common locations
            \$paths = @(
                'C:\Python312\python.exe',
                'C:\Python311\python.exe',
                'C:\Python310\python.exe',
                'C:\Program Files\Python312\python.exe',
                'C:\Program Files\Python311\python.exe',
                'C:\Program Files\Python310\python.exe',
                \"\$env:LOCALAPPDATA\Programs\Python\Python312\python.exe\",
                \"\$env:LOCALAPPDATA\Programs\Python\Python311\python.exe\",
                \"\$env:LOCALAPPDATA\Programs\Python\Python310\python.exe\"
            )
            foreach (\$path in \$paths) {
                if (Test-Path \$path) {
                    Write-Output \$path
                    break
                }
            }
        }
    }" 2>/dev/null | tr -d '\r' | head -1)

    if [ -z "$WIN_PYTHON" ]; then
        print_error "Python not found on Windows. Please install Python 3.10+ from https://python.org"
        print_info "After installing Python, run this script again or install manually with:"
        print_info "  pip install lemonade-sdk[oga-ryzenai] --extra-index-url=https://pypi.amd.com/simple"
        return 1
    fi

    print_success "Found Windows Python at: $WIN_PYTHON"

    # Check Python version
    # Get the full version output first
    PYTHON_VERSION_OUTPUT=$(powershell.exe -Command "& '$WIN_PYTHON' --version" 2>&1 | tr -d '\r')
    # Extract version number (handles "Python 3.10.0" format)
    PYTHON_VERSION=$(echo "$PYTHON_VERSION_OUTPUT" | grep -oE '3\.[0-9]+(\.[0-9]+)?' | head -1)

    if [ -z "$PYTHON_VERSION" ]; then
        print_warning "Could not determine Python version from: $PYTHON_VERSION_OUTPUT"
    else
        print_info "Python version: $PYTHON_VERSION"
        # Check if version is 3.10+
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
            print_error "Python 3.10+ is required. Found Python $PYTHON_VERSION"
            return 1
        fi
    fi

    # Ask about AMD hardware
    echo ""
    read -p "Do you have AMD GPU/NPU hardware? (y/N): " has_amd < /dev/tty

    # Install Lemonade
    print_info "Installing Lemonade SDK on Windows..."

    if [[ "$has_amd" == "y" || "$has_amd" == "Y" ]]; then
        print_info "Installing Lemonade SDK with AMD optimizations..."
        print_info "This may take a few minutes..."

        if powershell.exe -Command "& '$WIN_PYTHON' -m pip install --upgrade pip; & '$WIN_PYTHON' -m pip install 'lemonade-sdk[oga-ryzenai]' --extra-index-url=https://pypi.amd.com/simple" 2>&1 | tee /tmp/lemonade_install.log; then
            INSTALL_SUCCESS=true
        else
            INSTALL_SUCCESS=false
        fi
    else
        print_info "Installing standard Lemonade SDK..."
        print_info "This may take a few minutes..."

        if powershell.exe -Command "& '$WIN_PYTHON' -m pip install --upgrade pip; & '$WIN_PYTHON' -m pip install 'lemonade-sdk[llm-oga]'" 2>&1 | tee /tmp/lemonade_install.log; then
            INSTALL_SUCCESS=true
        else
            INSTALL_SUCCESS=false
        fi
    fi

    if [ "$INSTALL_SUCCESS" = true ]; then
        print_success "Lemonade SDK installed successfully!"

        # Create start script on Windows desktop
        DESKTOP_PATH="/mnt/c/Users/$USER/Desktop"
        if [ -d "$DESKTOP_PATH" ]; then
            cat > "$DESKTOP_PATH/start_lemonade.bat" << 'EOF'
@echo off
echo ========================================
echo     Lemonade Server for Morphik
echo ========================================
echo.
echo Starting server on http://localhost:8020
echo.
echo IMPORTANT: Keep this window open while using Morphik
echo Press Ctrl+C to stop the server
echo.
lemonade-server-dev server --port 8020 --ctx-size 100000
pause
EOF
            print_success "Created start_lemonade.bat on your Windows Desktop"
            echo ""
            print_info "📝 Next steps:"
            print_info "   1. Double-click start_lemonade.bat on your Desktop to start Lemonade"
            print_info "   2. Return to Morphik and select Lemonade models in the UI"
            print_info "   3. Enjoy fully local inference!"
        else
            print_warning "Could not create desktop shortcut. Start Lemonade manually with:"
            print_info "  lemonade-server-dev server --port 8020 --ctx-size 100000"
        fi

        # Update morphik.toml if it exists
        if [ -f "morphik.toml" ]; then
            print_info "Updating morphik.toml for WSL Docker -> Windows Lemonade connection..."
            sed -i.bak 's|http://localhost:8020|http://host.docker.internal:8020|g' morphik.toml 2>/dev/null || true
            rm -f morphik.toml.bak
            print_success "Configuration updated to use host.docker.internal"
        fi

        return 0
    else
        print_error "Failed to install Lemonade SDK"
        print_info "Check /tmp/lemonade_install.log for details"
        print_info "You can try manual installation with:"
        if [[ "$has_amd" == "y" || "$has_amd" == "Y" ]]; then
            print_info "  pip install lemonade-sdk[oga-ryzenai] --extra-index-url=https://pypi.amd.com/simple"
        else
            print_info "  pip install lemonade-sdk[llm-oga]"
        fi
        return 1
    fi
}

# Run installation
install_lemonade
