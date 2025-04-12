#!/bin/bash
# filepath: /home/p0wden/Documents/IAResearchAgregator/SearchEngine/research-cli.sh

# Colors for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Initialize conda for this script
eval "$(conda shell.bash hook)"
conda activate IATracker || { echo -e "${RED}ERROR: Failed to activate conda environment 'IATracker'${NC}"; exit 1; }
echo -e "${GREEN}>> Conda environment 'IATracker' activated${NC}"

# ASCII Art Header
print_header() {
    echo -e "${CYAN}"
    echo "  _   _                      _   ____                               _     "
    echo " | \\ | | ___ _   _ _ __ __ _| | |  _ \\ ___  ___  ___  __ _ _ __ ___| |__  "
    echo " |  \\| |/ _ \\ | | | '__/ _\` | | | |_) / _ \\/ __|/ _ \\/ _\` | '__/ __| '_ \\ "
    echo " | |\\  |  __/ |_| | | | (_| | | |  _ <  __/\\__ \\  __/ (_| | | | (__| | | |"
    echo " |_| \\_|\\___|\\__,_|_|  \\__,_|_| |_| \\_\\___||___/\\___|\\__,_|_|  \\___|_| |_|"
    echo -e "${NC}"
    echo -e "${GREEN}[ NEURAL RESEARCH INTERFACE ]${NC}"
    echo ""
}

# Help message
show_help() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  research-cli.sh [options] [\"<your research question>\"]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  -g, --gui         Launch the graphical user interface (default)"
    echo "  -c, --cli         Run in command-line interface mode"
    echo "  -q, --quick       Use quick answer mode (no LLM)"
    echo "  -k, --top-k NUM   Number of papers to consider (default: 5)"
    echo "  -t, --temp NUM    Temperature for answer generation (default: 0.7)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  research-cli.sh                                 # Launch the GUI (default)"
    echo "  research-cli.sh -c \"How do attention mechanisms work?\""
    echo "  research-cli.sh -c --quick \"What is RLHF?\""
    echo "  research-cli.sh -c -k 8 \"Latest methods for LLM finetuning\""
    echo ""
}

# Default values
GUI_MODE=true  # GUI mode is now true by default
CLI_MODE=false
QUICK=false
TOP_K=5
TEMP=0.7
QUESTION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -g|--gui)
            GUI_MODE=true
            CLI_MODE=false
            shift
            ;;
        -c|--cli)
            CLI_MODE=true
            GUI_MODE=false
            shift
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift
            shift
            ;;
        -t|--temp)
            TEMP="$2"
            shift
            shift
            ;;
        -h|--help)
            print_header
            show_help
            exit 0
            ;;
        *)
            QUESTION="$1"
            # If we get a question, assume CLI mode unless explicitly set to GUI
            if [ "$GUI_MODE" = true ] && [ "$CLI_MODE" = false ]; then
                # Only force CLI mode if the user hasn't explicitly requested GUI mode
                CLI_MODE=true
                GUI_MODE=false
            fi
            shift
            ;;
    esac
done

# Print header
print_header

# Default to GUI mode unless CLI mode is specified
if [ "$CLI_MODE" = false ]; then
    echo -e "${YELLOW}>> LAUNCHING GRAPHICAL USER INTERFACE...${NC}"
    python3 $(dirname "$0")/ResearchGui.py
    exit 0
fi

# We're in CLI mode, so check if question is provided
if [ -z "$QUESTION" ]; then
    echo -e "${RED}ERROR: No question provided for CLI mode${NC}"
    show_help
    exit 1
fi

# Build Python command for CLI mode
PYTHON_CMD="python3 $(dirname "$0")/ResearchQuestionAnswerer.py --question \"$QUESTION\" --top_k $TOP_K --temperature $TEMP"

if [ "$QUICK" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --quick"
fi

# Show processing message
echo -e "${YELLOW}>> PROCESSING QUERY: ${QUESTION}${NC}"
echo -e "${YELLOW}>> DATA SOURCES: ${TOP_K} | ENTROPY: ${TEMP} | STEALTH MODE: ${QUICK}${NC}"
echo ""

# Execute the Python command
eval $PYTHON_CMD | sed "s/^/${GREEN}>> ${NC}/"

# Exit with success
exit 0