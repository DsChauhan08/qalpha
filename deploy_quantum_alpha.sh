#!/bin/bash
# Quantum Alpha V1 - Deployment Script
# Single-command deployment for backtesting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║             QUANTUM ALPHA V1 - DEPLOYMENT                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
MODE="backtest"
CAPITAL=100000
SYMBOLS="SPY"
STRATEGY="momentum"
VALIDATE=""
PAPER_BARS=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --capital)
            CAPITAL="$2"
            shift 2
            ;;
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --validate)
            VALIDATE="--validate"
            shift
            ;;
        --paper-bars)
            PAPER_BARS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./deploy_quantum_alpha.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode      backtest|paper|live (default: backtest)"
            echo "  --capital   Initial capital (default: 100000)"
            echo "  --symbols   Space-separated symbols (default: SPY)"
            echo "  --strategy  momentum|mean_reversion|composite"
            echo "  --validate  Run MCPT validation"
            echo "  --paper-bars  Number of recent bars for paper mode (default: 30)"
            echo ""
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
echo -e "${YELLOW}Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  Python version: ${GREEN}$PYTHON_VERSION${NC}"

# Check/create virtual environment
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Validate installation
echo -e "${YELLOW}Validating installation...${NC}"
python3 -c "import numpy; import pandas; import yfinance; print('  Core dependencies: OK')"

# Run based on mode
echo ""
echo -e "${GREEN}Running Quantum Alpha V1...${NC}"
echo "  Mode: $MODE"
echo "  Capital: \$$CAPITAL"
echo "  Symbols: $SYMBOLS"
echo "  Strategy: $STRATEGY"
echo ""

# Convert symbols to Python list format
SYMBOL_ARGS=""
for sym in $SYMBOLS; do
    SYMBOL_ARGS="$SYMBOL_ARGS $sym"
done

# Execute
python3 -m quantum_alpha.main \
    --mode "$MODE" \
    --capital "$CAPITAL" \
    --symbols $SYMBOL_ARGS \
    --strategy "$STRATEGY" \
    --paper-bars "$PAPER_BARS" \
    $VALIDATE

echo ""
echo -e "${GREEN}Execution complete.${NC}"
