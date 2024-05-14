# check for python3 installation, download if not exists
echo "checking for python installation"
command -v python3 >/dev/null 2>&1
if [ $? -ne 0 ]; then # python is installed
    echo "python not installed"
    # Function to check if a package manager is available
    check_package_manager() {
        if command -v $1 &> /dev/null; then
            PACKAGE_MANAGER=$1
        fi
    }

    # Check for package managers
    check_package_manager apt-get
    check_package_manager yum
    check_package_manager dnf
    check_package_manager brew

    # Install Python 3.12
    if [ -n "$PACKAGE_MANAGER" ]; then
        case "$PACKAGE_MANAGER" in
            "apt-get")
                sudo apt-get update
                sudo apt-get install -y software-properties-common
                sudo add-apt-repository ppa:deadsnakes/ppa
                sudo apt-get install -y python3.12
                ;;
            "yum" | "dnf")
                sudo yum install -y https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
                sudo yum install -y python3.12
                ;;
            "brew")
                brew install python@3.12
                ;;
            *)
                echo "Unsupported package manager. Please install Python 3.12 manually."
                ;;
        esac
        echo "installed python3.12"
    else
        echo "No supported package manager found. Please install Python 3.12 manually."
        exit
    fi
fi
echo "python is installed"
# check for venv, create venv if not exists
if [ -d "surveydata" ]; then
    echo "The 'surveydata' environment exists."
else #venv does not exist
    echo "The 'surveydata' environment exists."
    # Create the virtual environment
    python3 -m venv surveydata
fi
# check for packages in venv, install if not present
echo "installing relevant python packages"
pip3 install -r $(dirname $0)/requirements.txt
# launch app?
echo "starting app"
python3 $(dirname $0)/app.py
